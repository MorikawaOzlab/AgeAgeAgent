from __future__ import annotations

import html
import math
import os
import random
import re
import subprocess
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Pool, freeze_support
from pathlib import Path
from typing import Any, Iterable

# NumPy / BLAS が各プロセス内でさらにスレッドを増やすのを防ぐ。
# 必ず numpy/pandas import 前に設定する。
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

import numpy as np
import pandas as pd

from scml.std import SCML2024StdWorld, is_system_agent
from scml_agents import get_agents

from AgeAgeAgent import AgeAgeAgent

# ============================================================
# 設定
# ============================================================

SEED = 42

N_SIMULATIONS = 20
N_STEPS = 50

# SCML内の生産プロセスごとのエージェント数。
# 例: [4, 4, 4] なら level 0, 1, 2 にそれぞれ4体ずつ。
N_AGENTS_PER_PROCESS = [4, 4, 4]

# CPU側の並列プロセス数。
# Windowsでは多すぎるとspawn/メモリ負荷で逆に遅くなることがあるので、必要なら手で下げる。
N_JOBS = max(1, min(N_SIMULATIONS, (os.cpu_count() or 2) - 1))

# workerを何taskごとに再起動するか。
# 0: 再起動しない。CPU効率優先ならまず0推奨。
# メモリリーク/後半劣化が明確なら 4 や 8 を試す。
MAX_TASKS_PER_CHILD = 0

# True: 外生契約も契約数・取引量・価格に含める
# False: エージェント間の交渉契約だけを見る
INCLUDE_EXOGENOUS_CONTRACTS = True

# 契約情報をどのstepに紐づけるか。
# delivery_time: 実際に納品されるstep
# signed_at: 契約が成立したstep
CONTRACT_STEP_FIELD = "delivery_time"

OUTPUT_HTML = "scml_simulation_report.html"
OPEN_HTML_AFTER_RUN = True

WORLD_KWARGS: dict[str, Any] = {
    "construct_graphs": False,
    "compact": True,
    "no_logs": True,
    "fast": True,
}

STATE_METRICS = [
    "score",
    "inventory_penalized",
    "shortfall_penalty",
    "shortfall_quantity",
    "productivity",
]

TRADE_METRICS = [
    "buy_contracts",
    "sell_contracts",
    "buy_quantity",
    "sell_quantity",
]

PRICE_VALUE_COLUMNS = [
    "buy_price_value",
    "sell_price_value",
]

PRICE_QTY_COLUMNS = [
    "buy_price_quantity",
    "sell_price_quantity",
]

HELPER_PRICE_COLUMNS = PRICE_VALUE_COLUMNS + PRICE_QTY_COLUMNS

HTML_METRICS = STATE_METRICS + TRADE_METRICS + [
    "buy_avg_unit_price",
    "sell_avg_unit_price",
]

STEP_DISPLAY_COLUMNS = [
    "agent",
    "step",
    "level",
    *HTML_METRICS,
]

FINAL_LEVEL_DISPLAY_COLUMNS = [
    "level",
    *HTML_METRICS,
]

FINAL_AGENT_DISPLAY_COLUMNS = [
    "agent",
    "level",
    *HTML_METRICS,
]

# SVGに使う色。色指定はHTML生成用途なので固定でOK。
PALETTE = [
    "#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2",
    "#be123c", "#4d7c0f", "#9333ea", "#0f766e", "#b45309", "#1d4ed8",
]


@dataclass(frozen=True)
class SimulationTask:
    """1回分のシミュレーション設定。"""

    run_id: int
    seed: int
    agent_types: list[type]
    alias_by_class_name: dict[str, str]


# ============================================================
# Agent設定
# ============================================================


def build_agent_types() -> list[type]:
    """
    比較対象エージェントを作る。

    multiprocessing の子プロセスで不要な get_agents()/random.sample() を実行しないように、
    main() からだけ呼ぶ。
    """
    rng = random.Random(SEED)
    winners_2025 = get_agents(
        version=2025,
        track="std",
        winners_only=False,
        as_class=True,
    )
    return [AgeAgeAgent] + rng.sample(list(winners_2025), 7)


# ============================================================
# Alias / 配置
# ============================================================


def _agent_stem(agent_type: type) -> str:
    """クラス名から短縮名生成用の名前を作る。"""
    name = agent_type.__name__

    if name.endswith("Agent") and len(name) > len("Agent"):
        name = name[: -len("Agent")]

    name = re.sub(r"[^0-9a-zA-Z_]", "", name)

    if not name:
        name = "Ag"

    if name[0].isdigit():
        name = "A" + name

    return name


def make_agent_aliases(agent_types: list[type]) -> list[str]:
    """
    エージェント名を短縮する。

    基本:
        頭文字2文字

    2文字がかぶる場合:
        頭文字2文字 + 尻文字

    それでもかぶる場合:
        頭文字2文字 + 連番文字
    """
    stems = [_agent_stem(agent_type) for agent_type in agent_types]
    heads = [(stem + "XX")[:2] for stem in stems]

    used: set[str] = set()
    aliases: list[str] = []

    for stem, head in zip(stems, heads, strict=True):
        candidate = head if heads.count(head) == 1 else head + stem[-1]

        if candidate in used:
            for suffix in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
                candidate = head + suffix
                if candidate not in used:
                    break

        used.add(candidate)
        aliases.append(candidate)

    return aliases


def build_placement(
    agent_types: list[type],
    seed: int,
) -> tuple[list[type], list[int]]:
    """
    エージェント配置を作る。

    runごとに配置をシャッフルして、配置違いのシミュレーションを行う。
    """
    rng = random.Random(seed)
    n_slots = sum(N_AGENTS_PER_PROCESS)

    placed_types = [agent_types[i % len(agent_types)] for i in range(n_slots)]
    rng.shuffle(placed_types)

    agent_processes: list[int] = []
    for process, n_agents in enumerate(N_AGENTS_PER_PROCESS):
        agent_processes.extend([process] * n_agents)

    return placed_types, agent_processes


# ============================================================
# 安全な値取得
# ============================================================


def safe_float(value: Any, default: float = math.nan) -> float:
    """値をfloatに変換する。失敗したらdefaultを返す。"""
    try:
        return float(value)
    except Exception:
        return default


def finite_or_none(value: Any) -> float | None:
    """有限なfloatなら返し、それ以外はNoneを返す。"""
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def find_stat_value(
    stats: pd.DataFrame,
    step: int,
    agent_id: str,
    metric: str,
) -> float:
    """
    world.stats_df から値を安全に取る。

    基本は {metric}_{agent_id} を見る。
    念のため typo や別名にも対応する。
    """
    aliases = {
        "inventory_penalized": [
            "inventory_penalized",
            "inventory_ppenalized",
        ],
        "productivity": [
            "productivity",
            "productibity",
            "_productivity",
        ],
    }

    candidates = aliases.get(metric, [metric])

    for candidate in candidates:
        column = f"{candidate}_{agent_id}"
        if column in stats.columns:
            return safe_float(stats.loc[step, column])

    return math.nan


def extract_level_from_agent_id(agent_id: str) -> int:
    """
    agent_id から level を取り出す。

    SCMLのagent_idは末尾が @0, @1, @2 のようになることが多い。
    """
    match = re.search(r"@(\d+)$", str(agent_id))
    if match:
        return int(match.group(1))

    return -1


# ============================================================
# 契約集計
# ============================================================


def safe_contracts_df(world: SCML2024StdWorld) -> pd.DataFrame:
    """world.contracts_df を安全に取得する。"""
    try:
        contracts = world.contracts_df.copy()
    except Exception:
        return pd.DataFrame()

    if contracts.empty:
        return contracts

    required_columns = {
        "seller",
        "buyer",
        "quantity",
        "unit_price",
        CONTRACT_STEP_FIELD,
    }

    if not required_columns.issubset(contracts.columns):
        missing = sorted(required_columns.difference(contracts.columns))
        raise RuntimeError(f"contracts_df に必要な列がありません: {missing}")

    if "signed_at" in contracts.columns:
        contracts = contracts[contracts["signed_at"] >= 0]

    if "nullified_at" in contracts.columns:
        contracts = contracts[contracts["nullified_at"] < 0]

    if not INCLUDE_EXOGENOUS_CONTRACTS:
        contracts = contracts[
            ~contracts["seller"].map(is_system_agent)
            & ~contracts["buyer"].map(is_system_agent)
        ]

    return contracts


def collect_contract_metrics(
    world: SCML2024StdWorld,
    agent_ids: list[str],
) -> dict[tuple[int, str], dict[str, float]]:
    """step, agent_id ごとの売買別契約情報を集計する。"""
    metrics: dict[tuple[int, str], dict[str, float]] = {
        (step, agent_id): {
            "buy_contracts": 0.0,
            "sell_contracts": 0.0,
            "buy_quantity": 0.0,
            "sell_quantity": 0.0,
            "buy_price_value": 0.0,
            "sell_price_value": 0.0,
            "buy_price_quantity": 0.0,
            "sell_price_quantity": 0.0,
        }
        for step in range(world.n_steps)
        for agent_id in agent_ids
    }

    contracts = safe_contracts_df(world)
    if contracts.empty:
        return metrics

    agent_id_set = set(agent_ids)

    for _, contract in contracts.iterrows():
        step = int(safe_float(contract[CONTRACT_STEP_FIELD], default=-1))
        quantity = safe_float(contract["quantity"], default=0.0)
        unit_price = safe_float(contract["unit_price"], default=math.nan)

        if step < 0 or step >= world.n_steps:
            continue

        if quantity <= 0 or math.isnan(unit_price):
            continue

        seller = str(contract["seller"])
        buyer = str(contract["buyer"])
        price_value = quantity * unit_price

        if seller in agent_id_set:
            key = (step, seller)
            metrics[key]["sell_contracts"] += 1.0
            metrics[key]["sell_quantity"] += quantity
            metrics[key]["sell_price_value"] += price_value
            metrics[key]["sell_price_quantity"] += quantity

        if buyer in agent_id_set:
            key = (step, buyer)
            metrics[key]["buy_contracts"] += 1.0
            metrics[key]["buy_quantity"] += quantity
            metrics[key]["buy_price_value"] += price_value
            metrics[key]["buy_price_quantity"] += quantity

    return metrics


# ============================================================
# World実行・レコード化
# ============================================================


def get_controller_class_name(agent: Any) -> str:
    """
    SCMLのAdapterに包まれている場合は中身のcontroller名を取る。
    そうでなければagent自身のクラス名を取る。
    """
    controller = getattr(agent, "_obj", agent)
    return controller.__class__.__name__


def extract_run_records(
    world: SCML2024StdWorld,
    run_id: int,
    alias_by_class_name: dict[str, str],
) -> list[dict[str, Any]]:
    """1回のworldから step別・agent別・level別のレコードを取り出す。"""
    stats = world.stats_df.copy()
    agent_ids = [
        agent_id
        for agent_id in world.agents.keys()
        if not is_system_agent(agent_id)
    ]

    agent_alias: dict[str, str] = {}
    agent_level: dict[str, int] = {}

    for agent_id in agent_ids:
        class_name = get_controller_class_name(world.agents[agent_id])
        agent_alias[agent_id] = alias_by_class_name.get(class_name, class_name)
        agent_level[agent_id] = extract_level_from_agent_id(agent_id)

    contract_metrics = collect_contract_metrics(world, agent_ids)

    rows: list[dict[str, Any]] = []

    for step in range(world.n_steps):
        for agent_id in agent_ids:
            row: dict[str, Any] = {
                "run_id": run_id,
                "step": step,
                "level": agent_level[agent_id],
                "agent": agent_alias[agent_id],
                "agent_id": agent_id,
            }

            for metric in STATE_METRICS:
                row[metric] = find_stat_value(
                    stats=stats,
                    step=step,
                    agent_id=agent_id,
                    metric=metric,
                )

            row.update(contract_metrics[(step, agent_id)])
            rows.append(row)

    df = pd.DataFrame(rows)

    # 同じ種類のAgentが同じlevelに複数体いる場合、agent aliasごとにまとめる。
    grouped = df.groupby(
        ["run_id", "step", "level", "agent"],
        as_index=False,
    ).agg(
        {
            **{metric: "mean" for metric in STATE_METRICS},
            **{metric: "mean" for metric in TRADE_METRICS},
            **{metric: "sum" for metric in HELPER_PRICE_COLUMNS},
        }
    )

    return grouped.to_dict(orient="records")


def run_one_simulation(task: SimulationTask) -> list[dict[str, Any]]:
    """1回分のSCML worldを生成・実行・集計する。"""
    random.seed(task.seed)
    np.random.seed(task.seed)

    placed_types, agent_processes = build_placement(
        agent_types=task.agent_types,
        seed=task.seed,
    )

    config = SCML2024StdWorld.generate(
        agent_types=placed_types,
        agent_processes=agent_processes,
        n_processes=len(N_AGENTS_PER_PROCESS),
        n_steps=N_STEPS,
        random_agent_types=False,
        name=f"std_sim_{task.run_id:03d}",
        **WORLD_KWARGS,
    )

    world = SCML2024StdWorld(**config)
    world.run()

    return extract_run_records(
        world=world,
        run_id=task.run_id,
        alias_by_class_name=task.alias_by_class_name,
    )


def run_simulations(tasks: list[SimulationTask]) -> pd.DataFrame:
    """
    複数simulationを並列実行する。

    設計:
        - multiprocessing.Pool.imap_unordered(..., chunksize=1) を使う。
        - chunksize=1により、空いたworkerに次のrunをすぐ渡す。
        - Futureを大量保持しないのでメモリも軽い。
        - MAX_TASKS_PER_CHILDは基本0。worker再起動はCPU効率を落とすことがあるため、必要時だけ使う。
    """
    if not tasks:
        return pd.DataFrame()

    workers = min(max(1, N_JOBS), len(tasks))
    all_rows: list[dict[str, Any]] = []
    started_at = time.perf_counter()

    if workers == 1:
        for done, task in enumerate(tasks, start=1):
            rows = run_one_simulation(task)
            all_rows.extend(rows)
            elapsed = time.perf_counter() - started_at
            print(f"finished {done}/{len(tasks)} | run={task.run_id} | elapsed={elapsed:.1f}s")
        return pd.DataFrame(all_rows)

    print(
        f"parallel settings: workers={workers}, "
        f"chunksize=1, "
        f"maxtasksperchild={MAX_TASKS_PER_CHILD or 'disabled'}"
    )

    maxtasksperchild = MAX_TASKS_PER_CHILD if MAX_TASKS_PER_CHILD > 0 else None

    with Pool(processes=workers, maxtasksperchild=maxtasksperchild) as pool:
        for done, rows in enumerate(pool.imap_unordered(run_one_simulation, tasks, chunksize=1), start=1):
            all_rows.extend(rows)
            elapsed = time.perf_counter() - started_at
            remaining = len(tasks) - done
            print(f"finished {done}/{len(tasks)} | remaining={remaining} | elapsed={elapsed:.1f}s")

    return pd.DataFrame(all_rows)


# ============================================================
# 集計
# ============================================================


def add_average_prices(df: pd.DataFrame) -> pd.DataFrame:
    """価格の合計値から数量加重平均価格を作る。"""
    df = df.copy()

    df["buy_avg_unit_price"] = np.where(
        df["buy_price_quantity"] > 0,
        df["buy_price_value"] / df["buy_price_quantity"],
        np.nan,
    )

    df["sell_avg_unit_price"] = np.where(
        df["sell_price_quantity"] > 0,
        df["sell_price_value"] / df["sell_price_quantity"],
        np.nan,
    )

    return df.drop(columns=HELPER_PRICE_COLUMNS)


def aggregate_step_average(all_records: pd.DataFrame) -> pd.DataFrame:
    """全シミュレーションの step ごとの平均値を作る。"""
    grouped = all_records.groupby(
        ["agent", "step", "level"],
        as_index=False,
    ).agg(
        {
            **{metric: "mean" for metric in STATE_METRICS},
            **{metric: "mean" for metric in TRADE_METRICS},
            **{metric: "sum" for metric in HELPER_PRICE_COLUMNS},
        }
    )

    grouped = add_average_prices(grouped)

    return grouped[STEP_DISPLAY_COLUMNS].sort_values(
        ["agent", "level", "step"],
        ignore_index=True,
    )


def aggregate_final_average_by_level(all_records: pd.DataFrame) -> pd.DataFrame:
    """全シミュレーションの最終結果を level ごとに平均する。"""
    all_records = all_records.sort_values(["run_id", "level", "agent", "step"])

    last_state_by_agent = all_records.groupby(
        ["run_id", "level", "agent"],
        as_index=False,
    ).tail(1)
    last_state_by_agent = last_state_by_agent[["run_id", "level", "agent", *STATE_METRICS]]

    last_state_by_level = last_state_by_agent.groupby(
        ["run_id", "level"],
        as_index=False,
    ).agg({metric: "mean" for metric in STATE_METRICS})

    trade_total_by_agent = all_records.groupby(
        ["run_id", "level", "agent"],
        as_index=False,
    ).agg(
        {
            **{metric: "sum" for metric in TRADE_METRICS},
            **{metric: "sum" for metric in HELPER_PRICE_COLUMNS},
        }
    )

    trade_total_by_level = trade_total_by_agent.groupby(
        ["run_id", "level"],
        as_index=False,
    ).agg(
        {
            **{metric: "mean" for metric in TRADE_METRICS},
            **{metric: "sum" for metric in HELPER_PRICE_COLUMNS},
        }
    )

    per_run_final = pd.merge(
        last_state_by_level,
        trade_total_by_level,
        on=["run_id", "level"],
        how="inner",
    )

    final = per_run_final.groupby("level", as_index=False).agg(
        {
            **{metric: "mean" for metric in STATE_METRICS},
            **{metric: "mean" for metric in TRADE_METRICS},
            **{metric: "sum" for metric in HELPER_PRICE_COLUMNS},
        }
    )

    final = add_average_prices(final)

    return final[FINAL_LEVEL_DISPLAY_COLUMNS].sort_values(
        "level",
        ignore_index=True,
    )


def aggregate_final_average_by_agent(all_records: pd.DataFrame) -> pd.DataFrame:
    """全シミュレーションの最終結果を agent, level ごとに平均する。"""
    all_records = all_records.sort_values(["run_id", "agent", "level", "step"])

    last_state = all_records.groupby(
        ["run_id", "agent", "level"],
        as_index=False,
    ).tail(1)
    last_state = last_state[["run_id", "agent", "level", *STATE_METRICS]]

    trade_total = all_records.groupby(
        ["run_id", "agent", "level"],
        as_index=False,
    ).agg(
        {
            **{metric: "sum" for metric in TRADE_METRICS},
            **{metric: "sum" for metric in HELPER_PRICE_COLUMNS},
        }
    )

    per_run_final = pd.merge(
        last_state,
        trade_total,
        on=["run_id", "agent", "level"],
        how="inner",
    )

    final = per_run_final.groupby(["agent", "level"], as_index=False).agg(
        {
            **{metric: "mean" for metric in STATE_METRICS},
            **{metric: "mean" for metric in TRADE_METRICS},
            **{metric: "sum" for metric in HELPER_PRICE_COLUMNS},
        }
    )

    final = add_average_prices(final)

    return final[FINAL_AGENT_DISPLAY_COLUMNS].sort_values(
        ["agent", "level"],
        ignore_index=True,
    )


# ============================================================
# HTML / SVG生成
# ============================================================


def e(value: Any) -> str:
    """HTML escape."""
    return html.escape(str(value), quote=True)


def fmt(value: Any, digits: int = 4) -> str:
    """表示用フォーマット。"""
    number = finite_or_none(value)
    if number is None:
        return ""
    if abs(number) >= 1000:
        return f"{number:,.2f}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def dataframe_to_html_table(
    df: pd.DataFrame,
    *,
    max_rows: int | None = None,
    title: str | None = None,
) -> str:
    """DataFrameをHTMLテーブルにする。"""
    view = df.copy()
    if max_rows is not None:
        view = view.head(max_rows)

    title_html = f"<h3>{e(title)}</h3>" if title else ""
    header = "".join(f"<th>{e(col)}</th>" for col in view.columns)
    body_rows: list[str] = []

    for _, row in view.iterrows():
        cells = []
        for col_index, col in enumerate(view.columns):
            value = row[col]
            text = fmt(value) if isinstance(value, (int, float, np.floating)) else e(value)
            align_class = "left" if col_index == 0 or col in {"agent"} else ""
            cells.append(f'<td class="{align_class}">{text}</td>')
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    omitted = ""
    if max_rows is not None and len(df) > max_rows:
        omitted = f'<p class="note">表示は先頭{max_rows}行のみ。全体は {len(df)} 行。</p>'

    return f"""
    {title_html}
    <div class="table-wrap">
      <table>
        <thead><tr>{header}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </div>
    {omitted}
    """


def svg_legend(series: list[dict[str, Any]]) -> str:
    return "".join(
        f'<span><i class="swatch" style="background:{e(item["color"])}"></i>{e(item["name"])}</span>'
        for item in series
    )


def svg_line_chart(
    labels: list[Any],
    series: list[dict[str, Any]],
    *,
    width: int | None = None,
    height: int = 340,
) -> str:
    """JSなしで表示できるSVG折れ線グラフを作る。"""
    if width is None:
        width = max(900, 90 + len(labels) * 24)

    pad_left = 72
    pad_right = 28
    pad_top = 24
    pad_bottom = 48
    inner_w = max(1, width - pad_left - pad_right)
    inner_h = max(1, height - pad_top - pad_bottom)

    values: list[float] = []
    for item in series:
        for value in item.get("values", []):
            number = finite_or_none(value)
            if number is not None:
                values.append(number)

    if not values or not labels:
        return f'<svg viewBox="0 0 {width} {height}" class="chart"><text x="24" y="42" fill="#6b7280">表示できるデータがありません</text></svg>'

    min_y = min(values)
    max_y = max(values)
    if min_y == max_y:
        min_y -= 1
        max_y += 1
    y_pad = (max_y - min_y) * 0.08
    min_y -= y_pad
    max_y += y_pad

    def x_at(index: int) -> float:
        if len(labels) <= 1:
            return pad_left + inner_w / 2
        return pad_left + (index / (len(labels) - 1)) * inner_w

    def y_at(value: float) -> float:
        return pad_top + (1 - (value - min_y) / (max_y - min_y)) * inner_h

    parts: list[str] = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']

    # grid / y-axis labels
    for i in range(6):
        y = pad_top + (i / 5) * inner_h
        v = max_y - (i / 5) * (max_y - min_y)
        parts.append(f'<line x1="{pad_left}" y1="{y:.2f}" x2="{width - pad_right}" y2="{y:.2f}" class="gridline" />')
        parts.append(f'<text x="{pad_left - 8}" y="{y + 4:.2f}" class="axis-label" text-anchor="end">{e(fmt(v, 2))}</text>')

    # x-axis labels
    tick_every = max(1, math.ceil(len(labels) / 14))
    for i, label in enumerate(labels):
        if i % tick_every != 0 and i != len(labels) - 1:
            continue
        x = x_at(i)
        parts.append(f'<text x="{x:.2f}" y="{height - 18}" class="axis-label" text-anchor="middle">{e(label)}</text>')

    # lines
    for item in series:
        points: list[str] = []
        for i, value in enumerate(item.get("values", [])):
            number = finite_or_none(value)
            if number is None:
                continue
            points.append(f'{x_at(i):.2f},{y_at(number):.2f}')
        if len(points) >= 2:
            parts.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{e(item["color"])}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" />'
            )
        elif len(points) == 1:
            x, y = points[0].split(",")
            parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{e(item["color"])}" />')

    parts.append("</svg>")
    return "".join(parts)


def svg_bar_chart(
    labels: list[Any],
    values: list[Any],
    *,
    width: int | None = None,
    height: int = 300,
) -> str:
    """JSなしで表示できるSVG棒グラフを作る。"""
    if width is None:
        width = max(720, 120 + len(labels) * 90)

    pad_left = 72
    pad_right = 28
    pad_top = 24
    pad_bottom = 52
    inner_w = max(1, width - pad_left - pad_right)
    inner_h = max(1, height - pad_top - pad_bottom)

    nums = [finite_or_none(v) for v in values]
    finite_values = [v for v in nums if v is not None]
    if not finite_values:
        return f'<svg viewBox="0 0 {width} {height}" class="chart small-chart"><text x="24" y="42" fill="#6b7280">表示できるデータがありません</text></svg>'

    min_y = min(0.0, min(finite_values))
    max_y = max(finite_values)
    if max_y == min_y:
        max_y += 1

    def y_at(value: float) -> float:
        return pad_top + (1 - (value - min_y) / (max_y - min_y)) * inner_h

    zero_y = y_at(0)
    gap = 18
    bar_w = max(22, (inner_w - gap * (len(labels) - 1)) / max(1, len(labels)))

    parts: list[str] = [f'<svg viewBox="0 0 {width} {height}" class="chart small-chart" role="img">']

    for i in range(6):
        y = pad_top + (i / 5) * inner_h
        v = max_y - (i / 5) * (max_y - min_y)
        parts.append(f'<line x1="{pad_left}" y1="{y:.2f}" x2="{width - pad_right}" y2="{y:.2f}" class="gridline" />')
        parts.append(f'<text x="{pad_left - 8}" y="{y + 4:.2f}" class="axis-label" text-anchor="end">{e(fmt(v, 2))}</text>')

    for i, (label, value) in enumerate(zip(labels, nums, strict=False)):
        if value is None:
            continue
        x = pad_left + i * (bar_w + gap)
        y = y_at(value)
        h = abs(zero_y - y)
        color = PALETTE[i % len(PALETTE)]
        parts.append(f'<rect x="{x:.2f}" y="{min(y, zero_y):.2f}" width="{bar_w:.2f}" height="{h:.2f}" rx="5" fill="{color}" />')
        parts.append(f'<text x="{x + bar_w / 2:.2f}" y="{height - 22}" class="axis-label" text-anchor="middle">{e(label)}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.2f}" y="{min(y, zero_y) - 6:.2f}" class="value-label" text-anchor="middle">{e(fmt(value, 2))}</text>')

    parts.append("</svg>")
    return "".join(parts)


def make_step_series_for_agent(agent_df: pd.DataFrame, metric: str) -> tuple[list[Any], list[dict[str, Any]]]:
    """1agent分のstep系列をlevel別seriesにする。"""
    steps = sorted(agent_df["step"].dropna().unique(), key=lambda x: int(x))
    levels = sorted(agent_df["level"].dropna().unique(), key=lambda x: int(x))

    series: list[dict[str, Any]] = []
    for index, level in enumerate(levels):
        level_df = agent_df[agent_df["level"] == level]
        value_by_step = {int(row["step"]): row[metric] for _, row in level_df.iterrows()}
        series.append(
            {
                "name": f"level {int(level)}",
                "color": PALETTE[index % len(PALETTE)],
                "values": [value_by_step.get(int(step)) for step in steps],
            }
        )

    return steps, series


def build_agent_step_sections(step_average: pd.DataFrame) -> str:
    """stepごとのグラフをagentごとにまとめて生成する。"""
    sections: list[str] = []
    key_metrics = [
        "score",
        "productivity",
        "inventory_penalized",
        "shortfall_penalty",
        "shortfall_quantity",
        "buy_quantity",
        "sell_quantity",
        "buy_avg_unit_price",
        "sell_avg_unit_price",
    ]

    for agent, agent_df in step_average.groupby("agent", sort=True):
        cards: list[str] = []
        for metric in key_metrics:
            steps, series = make_step_series_for_agent(agent_df, metric)
            chart = svg_line_chart(steps, series, height=330)
            cards.append(
                f"""
                <details class="metric-detail" {'open' if metric in {'score', 'buy_quantity', 'sell_quantity'} else ''}>
                  <summary>{e(metric)}</summary>
                  <div class="chart-scroll">{chart}</div>
                  <div class="legend">{svg_legend(series)}</div>
                </details>
                """
            )

        sections.append(
            f"""
            <section class="card agent-section" id="agent-{e(agent)}">
              <h2>Agent: <span class="pill">{e(agent)}</span></h2>
              <p class="note">stepごとの平均推移。線はlevel別です。</p>
              {''.join(cards)}
            </section>
            """
        )

    return "".join(sections)


def build_final_level_charts(final_by_level: pd.DataFrame) -> str:
    """level別最終結果のグラフを生成する。"""
    labels = [f"L{int(level)}" for level in final_by_level["level"].tolist()]
    metrics = [
        "score",
        "productivity",
        "inventory_penalized",
        "shortfall_penalty",
        "shortfall_quantity",
        "buy_quantity",
        "sell_quantity",
        "buy_avg_unit_price",
        "sell_avg_unit_price",
    ]

    cards: list[str] = []
    for metric in metrics:
        values = final_by_level[metric].tolist()
        chart = svg_bar_chart(labels, values)
        cards.append(
            f"""
            <details class="metric-detail" {'open' if metric == 'score' else ''}>
              <summary>{e(metric)}</summary>
              <div class="chart-scroll">{chart}</div>
            </details>
            """
        )

    return "".join(cards)


def build_kpi_cards(
    *,
    elapsed_seconds: float,
    all_records: pd.DataFrame,
    final_by_level: pd.DataFrame,
) -> str:
    """概要KPIを生成する。"""
    best_level_row = final_by_level.sort_values("score", ascending=False).head(1)
    best_level = ""
    best_score = ""
    if not best_level_row.empty:
        best_level = f"L{int(best_level_row.iloc[0]['level'])}"
        best_score = fmt(best_level_row.iloc[0]["score"])

    items = [
        ("simulations", N_SIMULATIONS),
        ("steps", N_STEPS),
        ("records", len(all_records)),
        ("elapsed", f"{elapsed_seconds:.1f}s"),
        ("workers", N_JOBS),
        ("best level", best_level),
        ("best level score", best_score),
        ("contract step", CONTRACT_STEP_FIELD),
    ]

    return "".join(
        f"""
        <div class="kpi">
          <div class="name">{e(name)}</div>
          <div class="value">{e(value)}</div>
        </div>
        """
        for name, value in items
    )


def build_settings_table(alias_by_class_name: dict[str, str]) -> str:
    settings = pd.DataFrame(
        [
            {"key": "N_SIMULATIONS", "value": N_SIMULATIONS},
            {"key": "N_STEPS", "value": N_STEPS},
            {"key": "N_AGENTS_PER_PROCESS", "value": str(N_AGENTS_PER_PROCESS)},
            {"key": "N_JOBS", "value": N_JOBS},
            {"key": "MAX_TASKS_PER_CHILD", "value": MAX_TASKS_PER_CHILD},
            {"key": "INCLUDE_EXOGENOUS_CONTRACTS", "value": INCLUDE_EXOGENOUS_CONTRACTS},
            {"key": "CONTRACT_STEP_FIELD", "value": CONTRACT_STEP_FIELD},
            {"key": "SEED", "value": SEED},
            {"key": "aliases", "value": str(alias_by_class_name)},
        ]
    )
    return dataframe_to_html_table(settings)


def build_html_report(
    *,
    all_records: pd.DataFrame,
    step_average: pd.DataFrame,
    final_by_level: pd.DataFrame,
    final_by_agent: pd.DataFrame,
    alias_by_class_name: dict[str, str],
    output_path: Path,
    elapsed_seconds: float,
) -> str:
    """静的HTMLレポートを生成する。JS依存を最小限にして壊れにくくする。"""
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    agent_sections = build_agent_step_sections(step_average)
    final_level_charts = build_final_level_charts(final_by_level)
    kpis = build_kpi_cards(
        elapsed_seconds=elapsed_seconds,
        all_records=all_records,
        final_by_level=final_by_level,
    )

    final_by_level_table = dataframe_to_html_table(final_by_level, title="Final average by level")
    final_by_agent_table = dataframe_to_html_table(final_by_agent, title="Final average by agent / level")
    step_table = dataframe_to_html_table(step_average, max_rows=500, title="Step average")
    settings_table = build_settings_table(alias_by_class_name)

    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SCML Simulation Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7fb;
      --card: #ffffff;
      --text: #111827;
      --muted: #6b7280;
      --line: #e5e7eb;
      --accent: #2563eb;
      --accent-soft: #dbeafe;
      --shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
      --radius: 18px;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); }}
    header {{ padding: 26px 28px 12px; }}
    main {{ padding: 14px 28px 36px; max-width: 1500px; margin: 0 auto; }}
    h1 {{ margin: 0 0 6px; font-size: 30px; letter-spacing: -0.04em; }}
    h2 {{ margin: 0 0 14px; font-size: 19px; }}
    h3 {{ margin: 0 0 10px; font-size: 16px; }}
    .subtitle {{ color: var(--muted); font-size: 14px; line-height: 1.6; }}
    .nav {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 14px 28px; }}
    .nav a {{ text-decoration: none; border: 1px solid var(--line); background: white; color: var(--text); border-radius: 999px; padding: 9px 14px; font-weight: 800; }}
    .nav a:hover {{ background: var(--accent); color: white; border-color: var(--accent); }}
    .grid {{ display: grid; gap: 16px; }}
    .grid-2 {{ grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow); padding: 18px; min-width: 0; margin-bottom: 16px; }}
    .kpis {{ display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr)); gap: 10px; }}
    .kpi {{ border: 1px solid var(--line); border-radius: 14px; padding: 12px; background: #fbfdff; }}
    .kpi .name {{ color: var(--muted); font-size: 12px; font-weight: 800; }}
    .kpi .value {{ font-size: 22px; font-weight: 900; margin-top: 4px; }}
    .note {{ color: var(--muted); font-size: 13px; line-height: 1.6; }}
    .pill {{ display: inline-flex; align-items: center; justify-content: center; min-width: 56px; padding: 4px 8px; border-radius: 999px; font-size: 13px; font-weight: 900; background: var(--accent-soft); color: var(--accent); }}
    .metric-detail {{ border: 1px solid var(--line); border-radius: 16px; padding: 10px 12px; margin: 10px 0; background: #fbfdff; }}
    .metric-detail > summary {{ cursor: pointer; font-weight: 900; color: #374151; }}
    .chart-scroll {{ width: 100%; overflow-x: auto; overflow-y: hidden; border: 1px solid var(--line); border-radius: 14px; background: white; padding: 8px; margin-top: 10px; }}
    .chart {{ display: block; height: auto; min-width: 780px; }}
    .small-chart {{ min-width: 680px; }}
    .gridline {{ stroke: #e5e7eb; stroke-width: 1; }}
    .axis-label {{ fill: #6b7280; font-size: 12px; }}
    .value-label {{ fill: #374151; font-size: 11px; font-weight: 800; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; font-size: 12px; color: var(--muted); }}
    .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
    .swatch {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 14px; max-height: 560px; background: white; }}
    table {{ border-collapse: collapse; width: 100%; min-width: 900px; background: white; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: right; font-size: 13px; white-space: nowrap; }}
    th {{ position: sticky; top: 0; background: #f8fafc; color: #374151; z-index: 1; }}
    th:first-child, td.left {{ text-align: left; }}
    tr:hover td {{ background: #f9fbff; }}
    @media (max-width: 900px) {{
      main, header {{ padding-left: 14px; padding-right: 14px; }}
      .nav {{ margin-left: 14px; margin-right: 14px; }}
      .grid-2 {{ grid-template-columns: 1fr; }}
      .kpis {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>SCML Simulation Report</h1>
    <div class="subtitle">
      generated: {e(generated_at)} / output: {e(output_path)}<br>
      HTMLは静的SVG中心です。Canvasや複雑なJSに依存しないので、Chromeで壊れにくい設計です。
    </div>
  </header>

  <nav class="nav">
    <a href="#overview">概要</a>
    <a href="#final-level">level別最終結果</a>
    <a href="#step-agent">agent別step推移</a>
    <a href="#tables">表</a>
  </nav>

  <main>
    <section id="overview" class="card">
      <h2>概要</h2>
      <div class="kpis">{kpis}</div>
    </section>

    <section class="grid grid-2">
      <div id="final-level" class="card">
        <h2>level別 最終結果グラフ</h2>
        <p class="note">最終結果はlevelごとに平均しています。</p>
        {final_level_charts}
      </div>
      <div class="card">
        <h2>実行設定</h2>
        {settings_table}
      </div>
    </section>

    <section id="step-agent">
      <h2>agent別 step推移</h2>
      <p class="note">stepごとの結果はagentごとにまとめています。各agent内でlevel別の線を表示します。</p>
      {agent_sections}
    </section>

    <section id="tables" class="card">
      <h2>集計表</h2>
      {final_by_level_table}
      <br>
      {final_by_agent_table}
      <br>
      {step_table}
    </section>
  </main>
</body>
</html>
"""


def generate_html_report(
    *,
    all_records: pd.DataFrame,
    step_average: pd.DataFrame,
    final_by_level: pd.DataFrame,
    final_by_agent: pd.DataFrame,
    alias_by_class_name: dict[str, str],
    output_path: str | Path = OUTPUT_HTML,
    elapsed_seconds: float = 0.0,
) -> Path:
    """HTMLファイルを生成する。"""
    path = Path(output_path).resolve()
    html_text = build_html_report(
        all_records=all_records,
        step_average=step_average,
        final_by_level=final_by_level,
        final_by_agent=final_by_agent,
        alias_by_class_name=alias_by_class_name,
        output_path=path,
        elapsed_seconds=elapsed_seconds,
    )
    path.write_text(html_text, encoding="utf-8")
    return path


def open_html_in_chrome(html_path: Path) -> bool:
    """HTMLをChromeで開く。Chromeが見つからなければ既定ブラウザで開く。"""
    url = html_path.resolve().as_uri()

    candidates: list[Path] = []
    for env_name in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        base = os.environ.get(env_name)
        if base:
            candidates.append(Path(base) / "Google" / "Chrome" / "Application" / "chrome.exe")

    for chrome_path in candidates:
        if chrome_path.exists():
            subprocess.Popen([str(chrome_path), url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True

    for browser_name in ("chrome", "google-chrome", "chromium", "chromium-browser"):
        try:
            controller = webbrowser.get(browser_name)
            controller.open(url)
            return True
        except webbrowser.Error:
            pass

    webbrowser.open(url)
    return False


# ============================================================
# print表示
# ============================================================


def print_dataframe(title: str, df: pd.DataFrame) -> None:
    """DataFrameを見やすくprintする。"""
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        240,
        "display.float_format",
        "{:.4f}".format,
    ):
        print(df.to_string(index=False))


def print_step_results_by_agent(step_average: pd.DataFrame) -> None:
    """stepごとの結果をagentごとに分けて表示する。"""
    for agent_name, agent_df in step_average.groupby("agent", sort=True):
        display_df = agent_df.drop(columns=["agent"]).sort_values(
            ["level", "step"],
            ignore_index=True,
        )

        print_dataframe(
            title=f"STEP AVERAGE OVER ALL SIMULATIONS - AGENT: {agent_name}",
            df=display_df,
        )


def print_final_results_by_level(final_average_by_level: pd.DataFrame) -> None:
    """最終結果をlevelごとに表示する。"""
    print_dataframe(
        title="FINAL AVERAGE OVER ALL SIMULATIONS - BY LEVEL",
        df=final_average_by_level,
    )


# ============================================================
# main
# ============================================================


def main() -> None:
    started_at = time.perf_counter()

    agent_types = build_agent_types()
    aliases = make_agent_aliases(agent_types)

    alias_by_class_name = {
        agent_type.__name__: alias
        for agent_type, alias in zip(agent_types, aliases, strict=True)
    }

    print("Agent aliases:")
    for agent_type, alias in zip(agent_types, aliases, strict=True):
        print(f"  {agent_type.__name__} -> {alias}")

    tasks = [
        SimulationTask(
            run_id=run_id,
            seed=SEED + run_id,
            agent_types=agent_types,
            alias_by_class_name=alias_by_class_name,
        )
        for run_id in range(N_SIMULATIONS)
    ]

    all_records = run_simulations(tasks)
    if all_records.empty:
        raise RuntimeError("シミュレーション結果が空です。")

    step_average = aggregate_step_average(all_records)
    final_average_by_level = aggregate_final_average_by_level(all_records)
    final_average_by_agent = aggregate_final_average_by_agent(all_records)

    print_step_results_by_agent(step_average)
    print_final_results_by_level(final_average_by_level)

    elapsed_seconds = time.perf_counter() - started_at
    html_path = generate_html_report(
        all_records=all_records,
        step_average=step_average,
        final_by_level=final_average_by_level,
        final_by_agent=final_average_by_agent,
        alias_by_class_name=alias_by_class_name,
        output_path=OUTPUT_HTML,
        elapsed_seconds=elapsed_seconds,
    )

    print(f"\nHTML report generated: {html_path}")

    if OPEN_HTML_AFTER_RUN:
        open_html_in_chrome(html_path)
        print("HTMLをブラウザで開きました。")


if __name__ == "__main__":
    freeze_support()
    main()
