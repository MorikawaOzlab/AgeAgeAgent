from __future__ import annotations

import os

# TensorFlow / oneDNN 系のログを少し静かにする
# scml_agents 側の import 前に設定しておく
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import re
import time
import random
import traceback
from collections import defaultdict
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scml.std import *
from scml_agents import get_agents

from AgeAgeAgent import AgeAgeAgent


# =========================
# 設定
# =========================

N_SIMULATIONS = 10
N_STEPS = 50

# CPUを使い切りすぎると不安定なら 4 などに固定する
MAX_WORKERS = min(N_SIMULATIONS, max(1, (os.cpu_count() or 2) - 1))

BASE_SEED = 20260528

TARGET_AGENTS = ["AS0", "AgeAgeAgent"]

N_PROCESSES = 3

# process は固定。types の順番だけ毎回シャッフルする
AGENT_PROCESSES = [0] * 4 + [1] * 5 + [2] * 5

# 比較に使う stats_df の項目だけ残す
# trading_price_ / sold_quantity_ / unit_price_ は product/market 別なので削除
STATS_SPECS = [
    ("score_", "Score", "score"),
    ("balance_", "Balance", "balance"),
    ("productivity_", "Productivity", "productivity"),
    ("shortfall_penalty_", "Shortfall Penalty", "penalty"),
    ("inventory_penalized_", "Inventory Penalized", "quantity"),
    ("inventory_input_", "Inventory Input", "quantity"),
    ("inventory_output_", "Inventory Output", "quantity"),
]

SHOW_COLUMN_MAPPING = True
SHOW_SHUFFLED_TYPES = True
SHOW_LOG_DIRS = True


# =========================
# 共通関数
# =========================

def format_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@lru_cache(maxsize=1)
def get_base_agent_types() -> list[type]:
    """
    シャッフル前の基本 types を作る。
    各プロセス内で1回だけ呼ばれる。
    """
    all_agents_2024 = get_agents(
        version=2024,
        track="std",
        winners_only=False,
        as_class=True,
    )
    all_agents_2025 = get_agents(
        version=2025,
        track="std",
        winners_only=False,
        as_class=True,
    )

    name_map_2024 = {cls.__name__: cls for cls in all_agents_2024}
    name_map_2025 = {cls.__name__: cls for cls in all_agents_2025}

    # 元のプログラムの types を維持
    types = [
        name_map_2025["AS0"],
        name_map_2025["KATSUDONAgent"],
        name_map_2025["PriceTrendStdAgent"],
        AgeAgeAgent,
        name_map_2025["XenoSotaAgent"],
        name_map_2024["PenguinAgent"],
        name_map_2024["AX"],
        AgeAgeAgent,
        name_map_2025["AS0"],
        name_map_2024["AX"],
        name_map_2024["MatchingPennies"],
        name_map_2025["AS0"],
        name_map_2024["CautiousStdAgent"],
        AgeAgeAgent,
    ]

    return types


def get_shuffled_agent_types(seed: int) -> list[type]:
    """
    シミュレーションごとに types の順番をシャッフルする。
    """
    rng = random.Random(seed)

    types = list(get_base_agent_types())
    rng.shuffle(types)

    return types


def get_class_name(agent_type: type) -> str:
    return getattr(agent_type, "__name__", str(agent_type))


def normalize_target_type_name(type_name: Any) -> str | None:
    """
    クラス名・agents.parquet の type などを、比較用の名前に正規化する。
    """
    type_name = str(type_name)

    if type_name == "AS0" or type_name.endswith(".AS0"):
        return "AS0"

    if "AgeAgeAgent" in type_name:
        return "AgeAgeAgent"

    return None


def get_target_name(agent_type: type) -> str | None:
    """
    types の各クラスが比較対象なら名前を返す。
    """
    return normalize_target_type_name(get_class_name(agent_type))


def extract_level(agent_name: str) -> int | None:
    """
    agent名の @ の後ろの数字を level として取り出す。

    例:
        00ASS0@0 -> 0
        08ASS0@1 -> 1
        13Ag@2 -> 2
    """
    match = re.search(r"@(\d+)$", str(agent_name))

    if match is None:
        return None

    return int(match.group(1))


def get_prefix_columns(stats_df: pd.DataFrame, prefix: str) -> list[str]:
    """
    元のグラフ関数と同じ方式で prefix から列を探索する。
    """
    return [str(col) for col in stats_df.columns if str(col).startswith(prefix)]


def get_suffix(col: str, prefix: str) -> str:
    return str(col)[len(prefix):]


# =========================
# stats_df 集計
# =========================

def infer_target_agent_ids_by_level(
    stats_df: pd.DataFrame,
    types: list[type],
) -> dict[str, dict[int, list[str]]]:
    """
    stats_df の score_ 列から、AS0 / AgeAgeAgent の実際の agent_id を level ごとに推定する。

    stats_df の列名例:
        score_00ASS0@0
        score_03Ag@0
        score_08ASS0@1
        score_13Ag@2

    先頭の 00, 03, 08, 13 は、シャッフル後の types の index に対応している想定。
    """
    score_cols = get_prefix_columns(stats_df, "score_")
    score_suffixes = [get_suffix(col, "score_") for col in score_cols]

    result: dict[str, dict[int, list[str]]] = {
        agent_name: defaultdict(list)
        for agent_name in TARGET_AGENTS
    }

    for index, agent_type in enumerate(types):
        target_name = get_target_name(agent_type)

        if target_name is None:
            continue

        index_prefix = f"{index:02d}"

        matched_agent_ids = [
            suffix
            for suffix in score_suffixes
            if suffix.startswith(index_prefix)
        ]

        for agent_id in matched_agent_ids:
            level = extract_level(agent_id)

            if level is None:
                continue

            result[target_name][level].append(agent_id)

    return {
        agent_name: dict(level_map)
        for agent_name, level_map in result.items()
    }


def select_agent_columns_by_level(
    stats_df: pd.DataFrame,
    prefix: str,
    agent_ids: list[str],
) -> list[str]:
    """
    score_ / balance_ / inventory_input_ などの agent 別列を選ぶ。
    """
    all_cols = get_prefix_columns(stats_df, prefix)
    agent_id_set = set(agent_ids)

    selected_cols = []

    for col in all_cols:
        suffix = get_suffix(col, prefix)

        if suffix in agent_id_set:
            selected_cols.append(col)

    return sorted(selected_cols)


def mean_of_columns(stats_df: pd.DataFrame, cols: list[str]) -> float:
    if not cols:
        return float("nan")

    values = pd.to_numeric(stats_df[cols].stack(), errors="coerce")
    return float(values.mean())


def get_all_levels(target_agent_ids_by_level: dict[str, dict[int, list[str]]]) -> list[int]:
    levels = set()

    for level_map in target_agent_ids_by_level.values():
        levels.update(level_map.keys())

    return sorted(levels)


def make_placement_records(
    run_id: int,
    target_agent_ids_by_level: dict[str, dict[int, list[str]]],
) -> list[dict[str, Any]]:
    """
    どの run で、どの agent_type が、どの level に存在したかを記録する。
    契約が0件でも、その level に存在したことを数えるために使う。
    """
    records = []

    for agent_type, level_map in target_agent_ids_by_level.items():
        for level, agent_ids in level_map.items():
            for agent_id in agent_ids:
                records.append(
                    {
                        "run_id": run_id,
                        "agent_type": agent_type,
                        "level": level,
                        "agent_name": agent_id,
                    }
                )

    return records


def format_types_by_level(types: list[type]) -> dict[int, list[str]]:
    """
    シャッフル後の types が各 level にどう配置されたか確認するための表示用。
    """
    result: dict[int, list[str]] = defaultdict(list)

    for index, agent_type in enumerate(types):
        level = AGENT_PROCESSES[index]
        result[level].append(f"{index:02d}:{get_class_name(agent_type)}")

    return dict(result)


# =========================
# parquet ログ探索
# =========================

def is_valid_log_dir(path: Path) -> bool:
    return (path / "agents.parquet").exists() and (path / "negs.parquet").exists()


def find_valid_log_dir_inside(path: Path) -> Path | None:
    """
    指定ディレクトリ以下から agents.parquet / negs.parquet が揃っている場所を探す。
    """
    if not path.exists():
        return None

    if path.is_file():
        path = path.parent

    if is_valid_log_dir(path):
        return path

    matches = []

    try:
        for agents_path in path.rglob("agents.parquet"):
            candidate = agents_path.parent

            if is_valid_log_dir(candidate):
                matches.append(candidate)
    except Exception:
        return None

    if not matches:
        return None

    # 最近更新されたものを使う
    matches.sort(
        key=lambda p: max(
            (p / "agents.parquet").stat().st_mtime,
            (p / "negs.parquet").stat().st_mtime,
        ),
        reverse=True,
    )

    return matches[0]


def get_possible_world_paths(world: Any, world_name: str) -> list[Path]:
    """
    world オブジェクトと一般的な negmas ログ場所から候補パスを作る。
    """
    candidates: list[Path] = []

    # world 側に log path 系属性があれば優先
    for attr in [
        "log_folder",
        "log_dir",
        "log_path",
        "folder",
        "path",
        "_log_folder",
        "_log_dir",
        "_log_path",
    ]:
        value = getattr(world, attr, None)

        if value is None:
            continue

        if callable(value):
            continue

        try:
            candidates.append(Path(value))
        except TypeError:
            pass

    # よくある保存場所
    home_logs = Path.home() / "negmas" / "logs"

    candidates.extend(
        [
            home_logs / world_name,
            home_logs,
            Path.cwd() / world_name,
            Path.cwd(),
        ]
    )

    # 重複除去
    unique_candidates = []
    seen = set()

    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)

        if key in seen:
            continue

        seen.add(key)
        unique_candidates.append(path)

    return unique_candidates


def find_world_log_dir(world: Any, world_name: str) -> Path | None:
    """
    今回の world に対応する parquet ログの場所を探す。
    """
    candidates = get_possible_world_paths(world, world_name)

    for candidate in candidates:
        log_dir = find_valid_log_dir_inside(candidate)

        if log_dir is None:
            continue

        # world_name が入っているパスを優先する
        if world_name in str(log_dir):
            return log_dir

    # world_name を含まないが parquet が見つかった場合の fallback
    for candidate in candidates:
        log_dir = find_valid_log_dir_inside(candidate)

        if log_dir is not None:
            return log_dir

    return None


# =========================
# parquet 契約集計
# =========================

def build_agent_maps(agents_df: pd.DataFrame) -> tuple[dict[str, str], dict[int, str]]:
    """
    agents.parquet から

        name -> target_type
        id   -> name

    の対応表を作る。
    """
    name_to_type: dict[str, str] = {}
    id_to_name: dict[int, str] = {}

    for _, row in agents_df.iterrows():
        agent_id = row.get("id")
        agent_name = row.get("name")
        agent_type = row.get("type")

        if pd.isna(agent_name):
            continue

        agent_name = str(agent_name)

        if not pd.isna(agent_id):
            try:
                id_to_name[int(agent_id)] = agent_name
            except ValueError:
                pass

        target_type = normalize_target_type_name(agent_type)

        if target_type is not None:
            name_to_type[agent_name] = target_type

    return name_to_type, id_to_name


def looks_like_agent_name_series(series: pd.Series) -> bool:
    """
    その列が 00ASS0@0 みたいな agent 名っぽいか判定する。
    """
    sample = series.dropna().astype(str).head(30)

    if sample.empty:
        return False

    return sample.str.contains(r"@\d+$", regex=True).any()


def find_agent_name_columns(negs_df: pd.DataFrame) -> tuple[str, str] | None:
    """
    negs.parquet の中から、契約参加者の名前が入っている2列を探す。

    環境やバージョンで列名が違う可能性があるので、候補を複数見る。
    """
    candidate_pairs = [
        ("agent0", "agent1"),
        ("agent_name0", "agent_name1"),
        ("agent0_name", "agent1_name"),
        ("agent_time0", "agent_time1"),
    ]

    for col0, col1 in candidate_pairs:
        if col0 not in negs_df.columns or col1 not in negs_df.columns:
            continue

        if looks_like_agent_name_series(negs_df[col0]) and looks_like_agent_name_series(negs_df[col1]):
            return col0, col1

    return None


def add_agent_name_columns(
    negs_df: pd.DataFrame,
    id_to_name: dict[int, str],
) -> pd.DataFrame:
    """
    negs.parquet に agent_name_0 / agent_name_1 を追加する。

    優先:
        agent0 / agent1 などの名前列

    fallback:
        agent0_id / agent1_id から agents.parquet を使って復元
    """
    df = negs_df.copy()

    name_cols = find_agent_name_columns(df)

    if name_cols is not None:
        col0, col1 = name_cols
        df["agent_name_0"] = df[col0].astype(str)
        df["agent_name_1"] = df[col1].astype(str)
        return df

    if "agent0_id" in df.columns and "agent1_id" in df.columns:
        df["agent_name_0"] = df["agent0_id"].map(
            lambda x: id_to_name.get(int(x)) if not pd.isna(x) else None
        )
        df["agent_name_1"] = df["agent1_id"].map(
            lambda x: id_to_name.get(int(x)) if not pd.isna(x) else None
        )
        return df

    raise ValueError(
        "negs.parquet から契約参加者の列を特定できませんでした。"
    )


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    return text in {"true", "1", "yes", "y"}


def judge_side(target_level: int, partner_level: int) -> str | None:
    """
    ユーザー定義の売買方向。

    buy:
        自分の level が相手より大きい。
        例: @1 が @0 から買う。

    sell:
        自分の level が相手より小さい。
        例: @1 が @2 に売る。
    """
    if target_level > partner_level:
        return "buy"

    if target_level < partner_level:
        return "sell"

    return None


def make_contract_records_from_log_dir(
    log_dir: Path,
    run_id: int,
) -> list[dict[str, Any]]:
    """
    agents.parquet と negs.parquet から、AS0 / AgeAgeAgent 視点の契約レコードを作る。
    """
    agents_path = log_dir / "agents.parquet"
    negs_path = log_dir / "negs.parquet"

    agents_df = pd.read_parquet(agents_path)
    negs_df = pd.read_parquet(negs_path)

    name_to_type, id_to_name = build_agent_maps(agents_df)

    negs_df = add_agent_name_columns(
        negs_df=negs_df,
        id_to_name=id_to_name,
    )

    df = negs_df.copy()

    if "has_agreement" in df.columns:
        df = df[df["has_agreement"].map(to_bool)]

    df = df.copy()

    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")

    df = df[
        df["quantity"].notna()
        & df["unit_price"].notna()
        & (df["quantity"] > 0)
        & (df["unit_price"] > 0)
    ]

    records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        agent0 = row["agent_name_0"]
        agent1 = row["agent_name_1"]

        if pd.isna(agent0) or pd.isna(agent1):
            continue

        agent0 = str(agent0)
        agent1 = str(agent1)

        level0 = extract_level(agent0)
        level1 = extract_level(agent1)

        if level0 is None or level1 is None:
            continue

        if level0 == level1:
            continue

        participants = [
            (agent0, level0, agent1, level1),
            (agent1, level1, agent0, level0),
        ]

        for target_name, target_level, partner_name, partner_level in participants:
            target_type = name_to_type.get(target_name)

            if target_type not in TARGET_AGENTS:
                continue

            side = judge_side(
                target_level=target_level,
                partner_level=partner_level,
            )

            if side is None:
                continue

            quantity = float(row["quantity"])
            unit_price = float(row["unit_price"])

            records.append(
                {
                    "run_id": run_id,
                    "agent_type": target_type,
                    "agent_name": target_name,
                    "level": target_level,
                    "side": side,
                    "partner_name": partner_name,
                    "partner_level": partner_level,
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "trade_value": quantity * unit_price,
                    "log_dir": str(log_dir),
                }
            )

    return records


def summarize_contract_records(
    contract_records: list[dict[str, Any]],
    placement_records: list[dict[str, Any]],
    n_successful_runs: int,
) -> pd.DataFrame:
    """
    全シミュレーションの契約レコードから、
    AS0 / AgeAgeAgent の level別・buy/sell別の平均取引量・価格を出す。
    """
    levels = list(range(N_PROCESSES))

    full_index = pd.MultiIndex.from_product(
        [TARGET_AGENTS, levels, ["buy", "sell"]],
        names=["agent_type", "level", "side"],
    )

    if contract_records:
        df = pd.DataFrame(contract_records)

        grouped = (
            df
            .groupby(["agent_type", "level", "side"], as_index=False)
            .agg(
                n_contracts=("quantity", "size"),
                total_quantity=("quantity", "sum"),
                avg_quantity_per_contract=("quantity", "mean"),
                avg_unit_price=("unit_price", "mean"),
                total_trade_value=("trade_value", "sum"),
                n_runs_with_contracts=("run_id", "nunique"),
            )
        )

        grouped["weighted_avg_unit_price"] = grouped.apply(
            lambda row: (
                row["total_trade_value"] / row["total_quantity"]
                if row["total_quantity"] > 0
                else float("nan")
            ),
            axis=1,
        )
    else:
        grouped = pd.DataFrame(
            columns=[
                "agent_type",
                "level",
                "side",
                "n_contracts",
                "total_quantity",
                "avg_quantity_per_contract",
                "avg_unit_price",
                "total_trade_value",
                "n_runs_with_contracts",
                "weighted_avg_unit_price",
            ]
        )

    summary = (
        grouped
        .set_index(["agent_type", "level", "side"])
        .reindex(full_index)
        .reset_index()
    )

    if placement_records:
        placement_df = pd.DataFrame(placement_records)

        placement_summary = (
            placement_df
            .groupby(["agent_type", "level"], as_index=False)
            .agg(
                n_agent_instances=("agent_name", "size"),
                n_runs_with_agent=("run_id", "nunique"),
            )
        )
    else:
        placement_summary = pd.DataFrame(
            columns=[
                "agent_type",
                "level",
                "n_agent_instances",
                "n_runs_with_agent",
            ]
        )

    summary = summary.merge(
        placement_summary,
        on=["agent_type", "level"],
        how="left",
    )

    fill_zero_cols = [
        "n_contracts",
        "total_quantity",
        "total_trade_value",
        "n_runs_with_contracts",
        "n_agent_instances",
        "n_runs_with_agent",
    ]

    for col in fill_zero_cols:
        if col in summary.columns:
            summary[col] = summary[col].fillna(0)

    summary["n_contracts"] = summary["n_contracts"].astype(int)
    summary["n_runs_with_contracts"] = summary["n_runs_with_contracts"].astype(int)
    summary["n_agent_instances"] = summary["n_agent_instances"].astype(int)
    summary["n_runs_with_agent"] = summary["n_runs_with_agent"].astype(int)

    # 全成功シミュレーションあたりの取引量
    summary["avg_quantity_per_successful_run"] = (
        summary["total_quantity"] / max(n_successful_runs, 1)
    )

    # その level に対象エージェントが存在した run あたりの取引量
    summary["avg_quantity_per_present_run"] = summary.apply(
        lambda row: (
            row["total_quantity"] / row["n_runs_with_agent"]
            if row["n_runs_with_agent"] > 0
            else float("nan")
        ),
        axis=1,
    )

    summary = summary[
        [
            "agent_type",
            "level",
            "side",
            "n_agent_instances",
            "n_runs_with_agent",
            "n_runs_with_contracts",
            "n_contracts",
            "total_quantity",
            "avg_quantity_per_contract",
            "avg_quantity_per_successful_run",
            "avg_quantity_per_present_run",
            "avg_unit_price",
            "weighted_avg_unit_price",
        ]
    ]

    return summary


# =========================
# 1シミュレーション
# =========================

def run_one_simulation(run_id: int, seed: int) -> dict[str, Any]:
    """
    1回分のシミュレーションを実行して、
    stats_df 集計と parquet 契約レコードを返す。
    """
    try:
        start_time = time.perf_counter()

        random.seed(seed)
        np.random.seed(seed)

        # ここでシミュレーションごとに types をシャッフル
        types = get_shuffled_agent_types(seed)

        world_name = f"ageage_compare_{run_id:03d}_{seed}"

        world = SCML2024StdWorld(
            **SCML2024StdWorld.generate(
                agent_types=types,
                agent_processes=AGENT_PROCESSES,
                n_processes=N_PROCESSES,
                n_steps=N_STEPS,
                # parquet / visualization 用ログが必要なので True
                construct_graphs=True,
                random_agent_types=False,
                name="test_world",
            )
        )

        world.init()

        for _ in range(world.n_steps):
            world.step()

        stats_df = world.stats_df

        target_agent_ids_by_level = infer_target_agent_ids_by_level(
            stats_df=stats_df,
            types=types,
        )

        levels = get_all_levels(target_agent_ids_by_level)

        record = {
            "run_id": run_id,
            "seed": seed,
            "seconds": time.perf_counter() - start_time,
        }

        column_mapping = {}
        shuffled_types_by_level = format_types_by_level(types)

        for prefix, title, _ylabel in STATS_SPECS:
            column_mapping[title] = {}

            for level in levels:
                column_mapping[title][level] = {}

                for agent_name in TARGET_AGENTS:
                    agent_ids = target_agent_ids_by_level.get(agent_name, {}).get(level, [])

                    cols = select_agent_columns_by_level(
                        stats_df=stats_df,
                        prefix=prefix,
                        agent_ids=agent_ids,
                    )

                    key = f"{title}__level{level}__{agent_name}"
                    record[key] = mean_of_columns(stats_df, cols)

                    column_mapping[title][level][agent_name] = {
                        "agent_ids": agent_ids,
                        "columns": cols,
                    }

        placement_records = make_placement_records(
            run_id=run_id,
            target_agent_ids_by_level=target_agent_ids_by_level,
        )

        log_dir = find_world_log_dir(
            world=world,
            world_name=world_name,
        )

        contract_records: list[dict[str, Any]] = []
        log_warning = None

        if log_dir is None:
            log_warning = (
                f"run_id={run_id}: agents.parquet / negs.parquet の場所を見つけられませんでした。"
            )
        else:
            try:
                contract_records = make_contract_records_from_log_dir(
                    log_dir=log_dir,
                    run_id=run_id,
                )
            except Exception:
                log_warning = traceback.format_exc()

        return {
            "ok": True,
            "record": record,
            "placement_records": placement_records,
            "contract_records": contract_records,
            "column_mapping": column_mapping,
            "shuffled_types_by_level": shuffled_types_by_level,
            "log_dir": str(log_dir) if log_dir is not None else None,
            "log_warning": log_warning,
            "error": None,
        }

    except Exception:
        return {
            "ok": False,
            "record": None,
            "placement_records": [],
            "contract_records": [],
            "column_mapping": None,
            "shuffled_types_by_level": None,
            "log_dir": None,
            "log_warning": None,
            "error": traceback.format_exc(),
        }


# =========================
# 表示
# =========================

def print_shuffled_types(shuffled_types_by_level: dict[int, list[str]] | None) -> None:
    """
    最初の成功シミュレーションのシャッフル結果を表示する。
    """
    if not SHOW_SHUFFLED_TYPES:
        return

    if not shuffled_types_by_level:
        return

    print("\n========== Example Shuffled Types ==========")

    for level in sorted(shuffled_types_by_level.keys()):
        print(f"level {level}:")
        for item in shuffled_types_by_level[level]:
            print(f"  {item}")


def print_column_mapping(column_mapping: dict | None) -> None:
    """
    最初の成功シミュレーションで、どの stats_df 列が使われたか確認用に表示する。
    """
    if not SHOW_COLUMN_MAPPING:
        return

    if not column_mapping:
        return

    print("\n========== Stats Column Mapping ==========")

    for _prefix, title, _ylabel in STATS_SPECS:
        print(f"\n[{title}]")

        level_map = column_mapping.get(title, {})

        for level in sorted(level_map.keys()):
            print(f"  level {level}")

            for agent_name in TARGET_AGENTS:
                info = level_map.get(level, {}).get(agent_name, {})
                agent_ids = info.get("agent_ids", [])
                cols = info.get("columns", [])

                print(f"    {agent_name}")
                print(f"      agent_ids: {agent_ids}")
                print(f"      columns  : {cols}")


def print_stats_summary(records: list[dict[str, Any]]) -> None:
    """
    全シミュレーション結果から level ごとの stats_df 平均を出して print する。
    """
    if not records:
        print("成功したシミュレーションがありません。")
        return

    df = pd.DataFrame(records)

    rows = []

    for _prefix, title, ylabel in STATS_SPECS:
        level_pattern = re.compile(rf"^{re.escape(title)}__level(\d+)__")

        levels = sorted(
            {
                int(match.group(1))
                for col in df.columns
                if (match := level_pattern.match(str(col))) is not None
            }
        )

        for level in levels:
            as0_col = f"{title}__level{level}__AS0"
            age_col = f"{title}__level{level}__AgeAgeAgent"

            as0_mean = df[as0_col].mean() if as0_col in df.columns else float("nan")
            age_mean = df[age_col].mean() if age_col in df.columns else float("nan")

            as0_n = df[as0_col].count() if as0_col in df.columns else 0
            age_n = df[age_col].count() if age_col in df.columns else 0

            rows.append(
                {
                    "metric": title,
                    "level": level,
                    "unit": ylabel,
                    "AS0_mean": as0_mean,
                    "AgeAgeAgent_mean": age_mean,
                    "AgeAgeAgent - AS0": age_mean - as0_mean,
                    "AS0_n": as0_n,
                    "AgeAgeAgent_n": age_n,
                }
            )

    summary_df = pd.DataFrame(rows)

    print("\n========== Stats Summary by Level ==========")
    print(f"Successful simulations: {len(records)}")
    print(summary_df.to_string(index=False))

    print("\n========== Time Summary ==========")
    print(f"Avg simulation time: {df['seconds'].mean():.4f} sec")
    print(f"Min simulation time: {df['seconds'].min():.4f} sec")
    print(f"Max simulation time: {df['seconds'].max():.4f} sec")


def print_contract_summary(
    contract_records: list[dict[str, Any]],
    placement_records: list[dict[str, Any]],
    n_successful_runs: int,
) -> None:
    """
    parquet 由来の契約集計を表示する。
    """
    summary_df = summarize_contract_records(
        contract_records=contract_records,
        placement_records=placement_records,
        n_successful_runs=n_successful_runs,
    )

    print("\n========== Contract Summary by Level and Side ==========")
    print("side=buy  : 自分の level が相手より大きい契約。例 @1 が @0 から買う。")
    print("side=sell : 自分の level が相手より小さい契約。例 @1 が @2 に売る。")
    print(summary_df.to_string(index=False))


def main() -> None:
    print("========== Simulation Settings ==========")
    print(f"N_SIMULATIONS: {N_SIMULATIONS}")
    print(f"N_STEPS: {N_STEPS}")
    print(f"MAX_WORKERS: {MAX_WORKERS}")
    print("SHUFFLE_TYPES: True")
    print("CONTRACT_SUMMARY_SOURCE: agents.parquet + negs.parquet")
    print("=========================================")

    start_all = time.perf_counter()

    records: list[dict[str, Any]] = []
    placement_records: list[dict[str, Any]] = []
    contract_records: list[dict[str, Any]] = []

    errors = []
    log_warnings = []
    log_dirs = []

    first_column_mapping = None
    first_shuffled_types_by_level = None

    futures = {}

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for run_id in range(N_SIMULATIONS):
            seed = BASE_SEED + run_id
            future = executor.submit(run_one_simulation, run_id, seed)
            futures[future] = run_id

        for i, future in enumerate(as_completed(futures), start=1):
            run_id = futures[future]

            try:
                payload = future.result()
            except Exception:
                payload = {
                    "ok": False,
                    "record": None,
                    "placement_records": [],
                    "contract_records": [],
                    "column_mapping": None,
                    "shuffled_types_by_level": None,
                    "log_dir": None,
                    "log_warning": None,
                    "error": traceback.format_exc(),
                }

            if payload["ok"]:
                records.append(payload["record"])
                placement_records.extend(payload["placement_records"])
                contract_records.extend(payload["contract_records"])

                if payload.get("log_dir") is not None:
                    log_dirs.append(payload["log_dir"])

                if payload.get("log_warning"):
                    log_warnings.append(
                        {
                            "run_id": run_id,
                            "warning": payload["log_warning"],
                        }
                    )

                if first_column_mapping is None:
                    first_column_mapping = payload["column_mapping"]

                if first_shuffled_types_by_level is None:
                    first_shuffled_types_by_level = payload["shuffled_types_by_level"]
            else:
                errors.append(
                    {
                        "run_id": run_id,
                        "error": payload["error"],
                    }
                )

            elapsed = time.perf_counter() - start_all
            avg_time = elapsed / i
            eta = avg_time * (N_SIMULATIONS - i)

            print(
                f"[{i}/{N_SIMULATIONS}] finished | "
                f"success: {len(records)} | "
                f"failed: {len(errors)} | "
                f"contracts: {len(contract_records)} | "
                f"elapsed: {format_time(elapsed)} | "
                f"ETA: {format_time(eta)}"
            )

    print_shuffled_types(first_shuffled_types_by_level)
    print_column_mapping(first_column_mapping)
    print_stats_summary(records)

    print_contract_summary(
        contract_records=contract_records,
        placement_records=placement_records,
        n_successful_runs=len(records),
    )

    if SHOW_LOG_DIRS and log_dirs:
        print("\n========== Example Log Dirs ==========")
        for path in sorted(set(log_dirs))[:10]:
            print(path)

    if log_warnings:
        print("\n========== Log Warnings ==========")
        print(f"Warnings: {len(log_warnings)}")
        print("最初の warning だけ表示します。")
        print(log_warnings[0]["warning"])

    if errors:
        print("\n========== Errors ==========")
        print(f"Failed simulations: {len(errors)}")
        print("最初のエラーだけ表示します。")
        print(errors[0]["error"])


if __name__ == "__main__":
    mp.freeze_support()
    main()