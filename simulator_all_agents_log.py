from __future__ import annotations

import os

# TensorFlow / oneDNN 系ログをなるべく抑制
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("ABSL_LOGGING_MIN_LEVEL", "3")

import gc
import re
import time
import random
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import multiprocessing as mp
from typing import Any

import numpy as np
import pandas as pd

from scml.std import *
from scml_agents import get_agents

from AgeAgeAgent import AgeAgeAgent


# =========================
# 設定
# =========================

N_SIMULATIONS = 300
N_STEPS = 50

MAX_WORKERS = min(
    N_SIMULATIONS,
    max(1, (os.cpu_count() or 2) - 1),
)

# この回数ごとに ProcessPoolExecutor を完全に作り直す
# 10なら、10シミュレーションごとに worker プロセスが全終了する
BATCH_SIZE = 3000
# BATCH_SIZE = 55*2

BASE_SEED = 20260528

# None にすると、シミュレーションに参加した全エージェントを集計・表示する。
# 例: ["AS0", "AgeAgeAgent"] に戻すと、その2種類だけ表示できる。
TARGET_AGENTS: list[str] | None = None

N_PROCESSES = 3

AGENT_PROCESSES = [0] * 5 + [1] * 6 + [2] * 6
# AGENT_PROCESSES = None

# stats_df から見る比較項目
# trading_price_ / sold_quantity_ / unit_price_ は market/product 別なので除外
STATS_SPECS = [
    ("score_", "Score", "score"),
    ("balance_", "Balance", "balance"),
    ("productivity_", "Productivity", "productivity"),
    ("shortfall_penalty_", "Shortfall Penalty", "penalty"),
    ("inventory_penalized_", "Inventory Penalized", "quantity"),
    ("inventory_input_", "Inventory Input", "quantity"),
    ("inventory_output_", "Inventory Output", "quantity"),
]


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
    各 worker プロセス内で1回だけ呼ばれる。
    executor を作り直すたびに worker も作り直されるので、
    キャッシュもそこで破棄される。
    """
    all_agents_2024 = get_agents(
        version=2024,
        track="std",
        winners_only=True,
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

    # return [
    #     name_map_2025["AS0"],
    #     name_map_2025["KATSUDONAgent"],
    #     name_map_2025["PriceTrendStdAgent"],
    #     AgeAgeAgent,
    #     name_map_2025["XenoSotaAgent"],
    #     name_map_2024["PenguinAgent"],
    #     name_map_2025["PriceTrendStdAgent"],
    #     AgeAgeAgent,
    #     name_map_2025["AS0"],
    #     name_map_2024["AX"],
    #     name_map_2024["MatchingPennies"],
    #     name_map_2025["AS0"],
    #     name_map_2024["CautiousStdAgent"],
    #     AgeAgeAgent,
    # ]

    # return [
    #     AgeAgeAgent,
    #     AgeAgeAgent,
    #     name_map_2025["AS0"],
    # ] + random.sample(all_agents_2024 + all_agents_2025, 11)
    agent_types = [AgeAgeAgent] + list(all_agents_2024) + list(all_agents_2025)
    # agent_types = agent_types + agent_types

    # agent_types = agent_types + random.sample(list(agents_2025), 2)
    # print(agent_types, len(agent_types))

    return agent_types

def get_shuffled_agent_types(seed: int) -> list[type]:
    rng = random.Random(seed)
    types = list(get_base_agent_types())
    rng.shuffle(types)
    return types


def normalize_agent_type_name(type_name: Any) -> str | None:
    """
    agent class / class名文字列 / world.saved_contracts 内の seller_type, buyer_type を
    表示用の短いエージェント名にそろえる。
    """
    if type_name is None:
        return None

    if isinstance(type_name, type):
        return type_name.__name__

    text = str(type_name).strip()

    if not text or text == "None":
        return None

    # "<class 'package.module.AgentName'>" 形式への対応
    class_match = re.match(r"^<class ['\"](.+)['\"]>$", text)
    if class_match:
        text = class_match.group(1)

    # "package.module.AgentName" 形式への対応
    if "." in text:
        text = text.split(".")[-1]

    # 念のため、残った余計な記号を除去
    text = text.strip("'\"> ")

    return text or None


def get_selected_agent_type_names(types: list[type]) -> list[str]:
    """
    今回のシミュレーションに参加しているエージェント型名を返す。
    TARGET_AGENTS が None なら全員、リストならその対象だけ。
    """
    all_names = sorted(
        {
            name
            for agent_type in types
            if (name := normalize_agent_type_name(agent_type)) is not None
        }
    )

    if TARGET_AGENTS is None:
        return all_names

    target_set = set(TARGET_AGENTS)
    return [name for name in all_names if name in target_set]


def extract_level(agent_name: str) -> int | None:
    match = re.search(r"@(\d+)$", str(agent_name))

    if match is None:
        return None

    return int(match.group(1))


def get_prefix_columns(stats_df: pd.DataFrame, prefix: str) -> list[str]:
    return [str(col) for col in stats_df.columns if str(col).startswith(prefix)]


def get_suffix(col: str, prefix: str) -> str:
    return str(col)[len(prefix):]



def get_attr_value(obj: Any, names: list[str]) -> Any:
    """
    SCML のバージョン差を吸収するために、候補名から値を取る。
    """
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return None


def get_agent_id_from_object(agent: Any) -> str | None:
    """
    world.agents の値が agent オブジェクトだった場合に、実際の agent id/name を取る。
    """
    value = get_attr_value(agent, ["id", "name", "agent_id", "aid"])

    if value is None:
        return None

    text = str(value).strip()
    return text or None


def add_agent_type_mapping(
    mapping: dict[str, str],
    agent_id: Any,
    agent_type: Any,
) -> None:
    """
    agent_id -> agent_type_name の対応を追加する。
    """
    if agent_id is None or agent_type is None:
        return

    agent_id_text = str(agent_id).strip()
    agent_type_name = normalize_agent_type_name(agent_type)

    if not agent_id_text or agent_type_name is None:
        return

    mapping[agent_id_text] = agent_type_name


def update_agent_type_mapping_from_agents_attr(
    mapping: dict[str, str],
    agents: Any,
) -> None:
    """
    world.agents / world._agents などから agent_id -> type_name を作る。
    dict/list の両方に対応する。
    """
    if agents is None:
        return

    if isinstance(agents, dict):
        for key, agent in agents.items():
            # key が agent id になっているケース
            add_agent_type_mapping(mapping, key, agent.__class__)

            # agent オブジェクト自身が id/name を持っているケース
            agent_id = get_agent_id_from_object(agent)
            add_agent_type_mapping(mapping, agent_id, agent.__class__)
        return

    if isinstance(agents, (list, tuple, set)):
        for agent in agents:
            agent_id = get_agent_id_from_object(agent)
            add_agent_type_mapping(mapping, agent_id, agent.__class__)


def update_agent_type_mapping_from_agent_types_attr(
    mapping: dict[str, str],
    agent_types: Any,
) -> None:
    """
    world.agent_types / world._agent_types が dict の場合への保険。
    """
    if agent_types is None:
        return

    if isinstance(agent_types, dict):
        for key, agent_type in agent_types.items():
            add_agent_type_mapping(mapping, key, agent_type)


def update_agent_type_mapping_from_contracts(
    mapping: dict[str, str],
    world: Any,
) -> None:
    """
    saved_contracts から agent_id -> type_name を補完する。
    契約したエージェントだけだが、Level2 が stats 側で拾えない問題の保険になる。
    """
    contracts = getattr(world, "saved_contracts", None)

    if not contracts:
        return

    for contract in contracts:
        if isinstance(contract, dict):
            seller_name = contract.get("seller_name")
            seller_type = contract.get("seller_type")
            buyer_name = contract.get("buyer_name")
            buyer_type = contract.get("buyer_type")
        else:
            seller_name = getattr(contract, "seller_name", None)
            seller_type = getattr(contract, "seller_type", None)
            buyer_name = getattr(contract, "buyer_name", None)
            buyer_type = getattr(contract, "buyer_type", None)

        add_agent_type_mapping(mapping, seller_name, seller_type)
        add_agent_type_mapping(mapping, buyer_name, buyer_type)


def build_agent_type_name_by_id(world: Any) -> dict[str, str]:
    """
    実際に world に生成された agent_id -> agent_type_name の対応を作る。

    これを使うことで、types の index と stats_df の列名が一致するという仮定を避ける。
    """
    mapping: dict[str, str] = {}

    for attr_name in ["agents", "_agents"]:
        update_agent_type_mapping_from_agents_attr(
            mapping,
            getattr(world, attr_name, None),
        )

    for attr_name in ["agent_types", "_agent_types"]:
        update_agent_type_mapping_from_agent_types_attr(
            mapping,
            getattr(world, attr_name, None),
        )

    update_agent_type_mapping_from_contracts(mapping, world)

    return mapping


# =========================
# stats_df 集計
# =========================

def infer_target_agent_ids_by_level(
    stats_df: pd.DataFrame,
    types: list[type],
    agent_type_name_by_id: dict[str, str] | None = None,
) -> dict[str, dict[int, list[str]]]:
    """
    stats_df の score_ 列から、agent_type -> level -> agent_id を作る。

    以前は types の index から agent_id の先頭番号を推定していたが、
    world.generate() 側の配置とズレると Level2 などを拾えなくなる。
    まず world から作った agent_id -> agent_type_name を使い、足りない分だけ旧方式で補完する。
    """
    score_cols = get_prefix_columns(stats_df, "score_")
    score_suffixes = [get_suffix(col, "score_") for col in score_cols]

    selected_names = set(get_selected_agent_type_names(types))
    result: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    assigned_agent_ids: set[str] = set()

    # 1. 実際の world から得た対応で集計する。
    if agent_type_name_by_id:
        for agent_id in score_suffixes:
            target_name = agent_type_name_by_id.get(agent_id)

            if target_name is None or target_name not in selected_names:
                continue

            level = extract_level(agent_id)

            if level is None:
                continue

            result[target_name][level].append(agent_id)
            assigned_agent_ids.add(agent_id)

    # 2. world から型名を取れなかった列だけ、旧方式で補完する。
    #    ただし、すでに対応付けできた agent_id は上書きしない。
    for index, agent_type in enumerate(types):
        target_name = normalize_agent_type_name(agent_type)

        if target_name is None or target_name not in selected_names:
            continue

        index_prefix = f"{index:02d}"

        matched_agent_ids = [
            suffix
            for suffix in score_suffixes
            if suffix not in assigned_agent_ids and suffix.startswith(index_prefix)
        ]

        for agent_id in matched_agent_ids:
            level = extract_level(agent_id)

            if level is None:
                continue

            result[target_name][level].append(agent_id)
            assigned_agent_ids.add(agent_id)

    return {
        agent_name: dict(level_map)
        for agent_name, level_map in result.items()
    }


def select_agent_columns_by_level(
    stats_df: pd.DataFrame,
    prefix: str,
    agent_ids: list[str],
) -> list[str]:
    all_cols = get_prefix_columns(stats_df, prefix)
    agent_id_set = set(agent_ids)

    return sorted(
        col
        for col in all_cols
        if get_suffix(col, prefix) in agent_id_set
    )


def mean_of_columns(stats_df: pd.DataFrame, cols: list[str]) -> float:
    if not cols:
        return float("nan")

    values = pd.to_numeric(stats_df[cols].stack(), errors="coerce")
    return float(values.mean())


def make_stats_values(
    stats_df: pd.DataFrame,
    types: list[type],
    agent_type_name_by_id: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    target_agent_ids_by_level = infer_target_agent_ids_by_level(
        stats_df=stats_df,
        types=types,
        agent_type_name_by_id=agent_type_name_by_id,
    )

    values = []

    selected_agent_type_names = get_selected_agent_type_names(types)

    for prefix, title, unit in STATS_SPECS:
        for agent_type in selected_agent_type_names:
            level_map = target_agent_ids_by_level.get(agent_type, {})

            for level, agent_ids in level_map.items():
                cols = select_agent_columns_by_level(
                    stats_df=stats_df,
                    prefix=prefix,
                    agent_ids=agent_ids,
                )

                values.append(
                    {
                        "level": level,
                        "metric": title,
                        "unit": unit,
                        "agent_type": agent_type,
                        "value": mean_of_columns(stats_df, cols),
                    }
                )

    return values


def make_agent_run_base(
    run_id: int,
    stats_df: pd.DataFrame,
    types: list[type],
    agent_type_name_by_id: dict[str, str] | None = None,
) -> dict[tuple[int, str, str, str], dict[str, Any]]:
    """
    1エージェント個体 × 1run × side の基準行を作る。
    契約0件のエージェントも平均の分母に含める。
    """
    target_agent_ids_by_level = infer_target_agent_ids_by_level(
        stats_df=stats_df,
        types=types,
        agent_type_name_by_id=agent_type_name_by_id,
    )

    base = {}

    for agent_type, level_map in target_agent_ids_by_level.items():
        for level, agent_names in level_map.items():
            for agent_name in agent_names:
                for side in ["buy", "sell"]:
                    key = (level, side, agent_type, agent_name)
                    base[key] = {
                        "run_id": run_id,
                        "level": level,
                        "side": side,
                        "agent_type": agent_type,
                        "agent_name": agent_name,
                        "contracts": 0.0,
                        "quantity": 0.0,
                        "trade_value": 0.0,
                    }

    return base


# =========================
# world.saved_contracts 集計
# =========================

def contracts_to_dataframe(world: Any) -> pd.DataFrame:
    contracts = getattr(world, "saved_contracts", None)

    if not contracts:
        return pd.DataFrame()

    return pd.DataFrame.from_records(contracts)


def make_trade_summaries_from_world(
    world: Any,
    run_id: int,
    stats_df: pd.DataFrame,
    types: list[type],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    生の契約一覧を親プロセスへ返さない。
    1シミュレーション内で必要な小さい集計だけ作って返す。
    """
    agent_type_name_by_id = build_agent_type_name_by_id(world)

    agent_run_map = make_agent_run_base(
        run_id=run_id,
        stats_df=stats_df,
        types=types,
        agent_type_name_by_id=agent_type_name_by_id,
    )

    contract_group_map: dict[tuple[int, str, str], dict[str, float]] = defaultdict(
        lambda: {
            "contract_count": 0.0,
            "quantity_sum": 0.0,
            "unit_price_sum": 0.0,
            "trade_value_sum": 0.0,
        }
    )

    contracts = contracts_to_dataframe(world)

    if contracts.empty:
        return list(agent_run_map.values()), []

    required_columns = {
        "seller_name",
        "buyer_name",
        "seller_type",
        "buyer_type",
        "quantity",
        "unit_price",
    }

    missing_columns = required_columns - set(contracts.columns)

    if missing_columns:
        raise ValueError(
            f"world.saved_contracts に必要な列がありません: {sorted(missing_columns)}"
        )

    contracts = contracts.copy()

    contracts["quantity"] = pd.to_numeric(contracts["quantity"], errors="coerce")
    contracts["unit_price"] = pd.to_numeric(contracts["unit_price"], errors="coerce")

    contracts = contracts[
        contracts["quantity"].notna()
        & contracts["unit_price"].notna()
        & (contracts["quantity"] > 0)
        & (contracts["unit_price"] > 0)
    ]

    def add_trade(
        *,
        agent_type: str,
        agent_name: str,
        level: int,
        side: str,
        quantity: float,
        unit_price: float,
    ) -> None:
        trade_value = quantity * unit_price
        agent_key = (level, side, agent_type, agent_name)

        if agent_key not in agent_run_map:
            agent_run_map[agent_key] = {
                "run_id": run_id,
                "level": level,
                "side": side,
                "agent_type": agent_type,
                "agent_name": agent_name,
                "contracts": 0.0,
                "quantity": 0.0,
                "trade_value": 0.0,
            }

        agent_run_map[agent_key]["contracts"] += 1.0
        agent_run_map[agent_key]["quantity"] += quantity
        agent_run_map[agent_key]["trade_value"] += trade_value

        group_key = (level, side, agent_type)
        contract_group_map[group_key]["contract_count"] += 1.0
        contract_group_map[group_key]["quantity_sum"] += quantity
        contract_group_map[group_key]["unit_price_sum"] += unit_price
        contract_group_map[group_key]["trade_value_sum"] += trade_value

    selected_agent_type_names = set(get_selected_agent_type_names(types))

    for _, row in contracts.iterrows():
        seller_name = str(row["seller_name"])
        buyer_name = str(row["buyer_name"])

        seller_level = extract_level(seller_name)
        buyer_level = extract_level(buyer_name)

        seller_target_type = normalize_agent_type_name(row["seller_type"])
        buyer_target_type = normalize_agent_type_name(row["buyer_type"])

        quantity = float(row["quantity"])
        unit_price = float(row["unit_price"])

        # seller 側: sell
        # @2 -> BUYER の外部契約も含む
        if (
            seller_target_type in selected_agent_type_names
            and seller_level is not None
        ):
            add_trade(
                agent_type=seller_target_type,
                agent_name=seller_name,
                level=seller_level,
                side="sell",
                quantity=quantity,
                unit_price=unit_price,
            )

        # buyer 側: buy
        # SELLER -> @0 の外部契約も含む
        if (
            buyer_target_type in selected_agent_type_names
            and buyer_level is not None
        ):
            add_trade(
                agent_type=buyer_target_type,
                agent_name=buyer_name,
                level=buyer_level,
                side="buy",
                quantity=quantity,
                unit_price=unit_price,
            )

    agent_run_rows = []

    for row in agent_run_map.values():
        quantity = row["quantity"]
        trade_value = row["trade_value"]

        row = dict(row)
        row["weighted_unit_price"] = (
            trade_value / quantity
            if quantity > 0
            else np.nan
        )
        agent_run_rows.append(row)

    contract_group_rows = []

    for (level, side, agent_type), row in contract_group_map.items():
        contract_group_rows.append(
            {
                "level": level,
                "side": side,
                "agent_type": agent_type,
                **row,
            }
        )

    return agent_run_rows, contract_group_rows


# =========================
# 1シミュレーション
# =========================

def run_one_simulation(run_id: int, seed: int) -> dict[str, Any]:
    try:
        start_time = time.perf_counter()

        random.seed(seed)
        np.random.seed(seed)

        types = get_shuffled_agent_types(seed)

        world = SCML2024StdWorld(
            **SCML2024StdWorld.generate(
                agent_types=types,
                agent_processes=AGENT_PROCESSES,
                n_processes=N_PROCESSES,
                n_steps=N_STEPS,
                construct_graphs=False,
                random_agent_types=False,
                name=f"test_world_{run_id}",
            )
        )

        world.init()

        for _ in range(world.n_steps):
            world.step()

        stats_df = world.stats_df

        agent_type_name_by_id = build_agent_type_name_by_id(world)

        stats_values = make_stats_values(
            stats_df=stats_df,
            types=types,
            agent_type_name_by_id=agent_type_name_by_id,
        )

        agent_run_rows, contract_group_rows = make_trade_summaries_from_world(
            world=world,
            run_id=run_id,
            stats_df=stats_df,
            types=types,
        )

        seconds = time.perf_counter() - start_time

        del world
        del stats_df
        gc.collect()

        return {
            "ok": True,
            "seconds": seconds,
            "stats_values": stats_values,
            "agent_run_rows": agent_run_rows,
            "contract_group_rows": contract_group_rows,
            "error": None,
        }

    except Exception:
        gc.collect()

        return {
            "ok": False,
            "seconds": 0.0,
            "stats_values": [],
            "agent_run_rows": [],
            "contract_group_rows": [],
            "error": traceback.format_exc(),
        }


# =========================
# 集計器
# =========================

def update_stats_accumulator(
    stats_acc: dict[tuple[int, str, str, str], dict[str, float]],
    stats_values: list[dict[str, Any]],
) -> None:
    for row in stats_values:
        value = row["value"]

        if pd.isna(value):
            continue

        key = (
            int(row["level"]),
            str(row["metric"]),
            str(row["unit"]),
            str(row["agent_type"]),
        )

        if key not in stats_acc:
            stats_acc[key] = {
                "sum": 0.0,
                "count": 0.0,
            }

        stats_acc[key]["sum"] += float(value)
        stats_acc[key]["count"] += 1.0


def update_trade_accumulators(
    agent_run_acc: dict[tuple[int, str, str], dict[str, float]],
    contract_group_acc: dict[tuple[int, str, str], dict[str, float]],
    agent_run_rows: list[dict[str, Any]],
    contract_group_rows: list[dict[str, Any]],
) -> None:
    seen_run_groups = set()

    for row in agent_run_rows:
        key = (
            int(row["level"]),
            str(row["side"]),
            str(row["agent_type"]),
        )

        if key not in agent_run_acc:
            agent_run_acc[key] = {
                "n_agent_runs": 0.0,
                "n_runs_with_agent": 0.0,
                "sum_contracts": 0.0,
                "sum_quantity": 0.0,
                "sum_trade_value": 0.0,
                "sum_weighted_unit_price": 0.0,
                "count_weighted_unit_price": 0.0,
            }

        agent_run_acc[key]["n_agent_runs"] += 1.0
        agent_run_acc[key]["sum_contracts"] += float(row["contracts"])
        agent_run_acc[key]["sum_quantity"] += float(row["quantity"])
        agent_run_acc[key]["sum_trade_value"] += float(row["trade_value"])

        weighted_price = row.get("weighted_unit_price", np.nan)

        if not pd.isna(weighted_price):
            agent_run_acc[key]["sum_weighted_unit_price"] += float(weighted_price)
            agent_run_acc[key]["count_weighted_unit_price"] += 1.0

        run_key = (
            int(row["run_id"]),
            int(row["level"]),
            str(row["side"]),
            str(row["agent_type"]),
        )

        seen_run_groups.add(run_key)

    for _run_id, level, side, agent_type in seen_run_groups:
        key = (level, side, agent_type)

        if key not in agent_run_acc:
            agent_run_acc[key] = {
                "n_agent_runs": 0.0,
                "n_runs_with_agent": 0.0,
                "sum_contracts": 0.0,
                "sum_quantity": 0.0,
                "sum_trade_value": 0.0,
                "sum_weighted_unit_price": 0.0,
                "count_weighted_unit_price": 0.0,
            }

        agent_run_acc[key]["n_runs_with_agent"] += 1.0

    for row in contract_group_rows:
        key = (
            int(row["level"]),
            str(row["side"]),
            str(row["agent_type"]),
        )

        if key not in contract_group_acc:
            contract_group_acc[key] = {
                "contract_count": 0.0,
                "quantity_sum": 0.0,
                "unit_price_sum": 0.0,
                "trade_value_sum": 0.0,
            }

        contract_group_acc[key]["contract_count"] += float(row["contract_count"])
        contract_group_acc[key]["quantity_sum"] += float(row["quantity_sum"])
        contract_group_acc[key]["unit_price_sum"] += float(row["unit_price_sum"])
        contract_group_acc[key]["trade_value_sum"] += float(row["trade_value_sum"])


def update_time_accumulator(
    time_acc: dict[str, float],
    seconds: float,
) -> None:
    time_acc["count"] += 1.0
    time_acc["sum"] += seconds

    if time_acc["min"] < 0 or seconds < time_acc["min"]:
        time_acc["min"] = seconds

    if seconds > time_acc["max"]:
        time_acc["max"] = seconds


# =========================
# 表示
# =========================

def print_stats_summary(
    stats_acc: dict[tuple[int, str, str, str], dict[str, float]],
    successful_count: int,
    time_acc: dict[str, float],
) -> None:
    rows = []

    agent_types = sorted({agent_type for *_rest, agent_type in stats_acc.keys()})

    for level in range(N_PROCESSES):
        for _prefix, metric, unit in STATS_SPECS:
            for agent_type in agent_types:
                key = (level, metric, unit, agent_type)
                data = stats_acc.get(key, {"sum": 0.0, "count": 0.0})

                mean = (
                    data["sum"] / data["count"]
                    if data["count"] > 0
                    else np.nan
                )

                rows.append(
                    {
                        "level": level,
                        "metric": metric,
                        "unit": unit,
                        "agent_type": agent_type,
                        "mean": mean,
                        "n": int(data["count"]),
                    }
                )

    summary_df = pd.DataFrame(rows)

    if not summary_df.empty:
        metric_order = {metric: index for index, (_prefix, metric, _unit) in enumerate(STATS_SPECS)}
        summary_df["_metric_order"] = summary_df["metric"].map(metric_order)
        summary_df = summary_df.sort_values(
            ["level", "_metric_order", "mean", "agent_type"],
            ascending=[True, True, False, True],
        ).drop(columns="_metric_order")

    print("\n========== Stats Summary by Level and Agent ==========")
    print(f"Successful simulations: {successful_count}")
    print(summary_df.to_string(index=False))

    if time_acc["count"] > 0:
        print("\n========== Time Summary ==========")
        print(f"Avg simulation time: {time_acc['sum'] / time_acc['count']:.4f} sec")
        print(f"Min simulation time: {time_acc['min']:.4f} sec")
        print(f"Max simulation time: {time_acc['max']:.4f} sec")


def print_contract_summary(
    agent_run_acc: dict[tuple[int, str, str], dict[str, float]],
    contract_group_acc: dict[tuple[int, str, str], dict[str, float]],
) -> None:
    rows = []

    agent_types = sorted(
        {agent_type for _level, _side, agent_type in agent_run_acc.keys()}
        | {agent_type for _level, _side, agent_type in contract_group_acc.keys()}
    )

    for level in range(N_PROCESSES):
        for side in ["buy", "sell"]:
            for agent_type in agent_types:
                key = (level, side, agent_type)

                agent_data = agent_run_acc.get(
                    key,
                    {
                        "n_agent_runs": 0.0,
                        "n_runs_with_agent": 0.0,
                        "sum_contracts": 0.0,
                        "sum_quantity": 0.0,
                        "sum_trade_value": 0.0,
                        "sum_weighted_unit_price": 0.0,
                        "count_weighted_unit_price": 0.0,
                    },
                )

                contract_data = contract_group_acc.get(
                    key,
                    {
                        "contract_count": 0.0,
                        "quantity_sum": 0.0,
                        "unit_price_sum": 0.0,
                        "trade_value_sum": 0.0,
                    },
                )

                n_agent_runs = agent_data["n_agent_runs"]
                contract_count = contract_data["contract_count"]
                quantity_sum = contract_data["quantity_sum"]

                avg_n_contracts_per_agent_run = (
                    agent_data["sum_contracts"] / n_agent_runs
                    if n_agent_runs > 0
                    else np.nan
                )

                avg_total_quantity_per_agent_run = (
                    agent_data["sum_quantity"] / n_agent_runs
                    if n_agent_runs > 0
                    else np.nan
                )

                avg_trade_value_per_agent_run = (
                    agent_data["sum_trade_value"] / n_agent_runs
                    if n_agent_runs > 0
                    else np.nan
                )

                avg_weighted_unit_price_per_agent_run = (
                    agent_data["sum_weighted_unit_price"]
                    / agent_data["count_weighted_unit_price"]
                    if agent_data["count_weighted_unit_price"] > 0
                    else np.nan
                )

                avg_quantity_per_contract = (
                    contract_data["quantity_sum"] / contract_count
                    if contract_count > 0
                    else np.nan
                )

                avg_unit_price_per_contract = (
                    contract_data["unit_price_sum"] / contract_count
                    if contract_count > 0
                    else np.nan
                )

                weighted_avg_unit_price = (
                    contract_data["trade_value_sum"] / quantity_sum
                    if quantity_sum > 0
                    else np.nan
                )

                rows.append(
                    {
                        "level": level,
                        "side": side,
                        "agent_type": agent_type,
                        "n_agent_runs": int(n_agent_runs),
                        "n_runs_with_agent": int(agent_data["n_runs_with_agent"]),
                        "avg_n_contracts_per_agent_run": avg_n_contracts_per_agent_run,
                        "avg_total_quantity_per_agent_run": avg_total_quantity_per_agent_run,
                        "avg_trade_value_per_agent_run": avg_trade_value_per_agent_run,
                        "avg_quantity_per_contract": avg_quantity_per_contract,
                        "avg_unit_price_per_contract": avg_unit_price_per_contract,
                        "weighted_avg_unit_price": weighted_avg_unit_price,
                        "avg_weighted_unit_price_per_agent_run": avg_weighted_unit_price_per_agent_run,
                    }
                )

    summary_df = pd.DataFrame(rows)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["level", "side", "avg_trade_value_per_agent_run", "agent_type"],
            ascending=[True, True, False, True],
        )

    print("\n========== Contract Summary by Level, Side, and Agent ==========")
    print("外部契約も含む。")
    print("契約数・数量・取引金額は 1エージェント・1シミュレーションあたりの平均。")
    print("side=buy  : buyer_name 側の契約。例 SELLER -> @0, @0 -> @1, @1 -> @2")
    print("side=sell : seller_name 側の契約。例 @0 -> @1, @1 -> @2, @2 -> BUYER")
    print(summary_df.to_string(index=False))


# =========================
# main
# =========================

def main() -> None:
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    print("========== Simulation Settings ==========")
    print(f"N_SIMULATIONS: {N_SIMULATIONS}")
    print(f"N_STEPS: {N_STEPS}")
    print(f"MAX_WORKERS: {MAX_WORKERS}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print("EXTERNAL_CONTRACTS: included")
    print("MEMORY_MODE: recreate executor every batch")
    print("=========================================")

    start_all = time.perf_counter()

    stats_acc: dict[tuple[int, str, str, str], dict[str, float]] = {}
    agent_run_acc: dict[tuple[int, str, str], dict[str, float]] = {}
    contract_group_acc: dict[tuple[int, str, str], dict[str, float]] = {}

    time_acc = {
        "count": 0.0,
        "sum": 0.0,
        "min": -1.0,
        "max": 0.0,
    }

    completed_count = 0
    successful_count = 0
    failed_count = 0
    first_error = None

    for batch_start in range(0, N_SIMULATIONS, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, N_SIMULATIONS)

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    run_one_simulation,
                    run_id,
                    BASE_SEED + run_id,
                ): run_id
                for run_id in range(batch_start, batch_end)
            }

            for future in as_completed(futures):
                run_id = futures[future]
                completed_count += 1

                try:
                    payload = future.result()
                except Exception:
                    payload = {
                        "ok": False,
                        "seconds": 0.0,
                        "stats_values": [],
                        "agent_run_rows": [],
                        "contract_group_rows": [],
                        "error": traceback.format_exc(),
                    }

                if payload["ok"]:
                    successful_count += 1

                    update_time_accumulator(
                        time_acc=time_acc,
                        seconds=float(payload["seconds"]),
                    )

                    update_stats_accumulator(
                        stats_acc=stats_acc,
                        stats_values=payload["stats_values"],
                    )

                    update_trade_accumulators(
                        agent_run_acc=agent_run_acc,
                        contract_group_acc=contract_group_acc,
                        agent_run_rows=payload["agent_run_rows"],
                        contract_group_rows=payload["contract_group_rows"],
                    )
                else:
                    failed_count += 1

                    if first_error is None:
                        first_error = {
                            "run_id": run_id,
                            "error": payload["error"],
                        }

                elapsed = time.perf_counter() - start_all
                avg_time = elapsed / completed_count
                eta = avg_time * (N_SIMULATIONS - completed_count)

                print(
                    f"[{completed_count}/{N_SIMULATIONS}] finished | "
                    f"success: {successful_count} | "
                    f"failed: {failed_count} | "
                    f"elapsed: {format_time(elapsed)} | "
                    f"ETA: {format_time(eta)}"
                )

                del payload

            futures.clear()

        # ここで executor が閉じられ、workerプロセスが全終了する
        gc.collect()

    print_stats_summary(
        stats_acc=stats_acc,
        successful_count=successful_count,
        time_acc=time_acc,
    )

    print_contract_summary(
        agent_run_acc=agent_run_acc,
        contract_group_acc=contract_group_acc,
    )

    if first_error is not None:
        print("\n========== Errors ==========")
        print(f"Failed simulations: {failed_count}")
        print("最初のエラーだけ表示します。")
        print(first_error["error"])


if __name__ == "__main__":
    mp.freeze_support()
    main()