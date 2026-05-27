from __future__ import annotations

from collections import defaultdict
from multiprocessing import freeze_support
from pathlib import Path
import random
import re
import string

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from negmas import ResponseType
from scml.std import *
from scml.utils import anac2024_std
from scml_agents import get_agents

from AgeAgeAgent import AgeAgeAgent


# =========================
# Settings
# =========================

RANDOM_SEED = 42

N_SAMPLED_WINNERS = 9

TOURNAMENT_PATH = r"C:\t"
ALIAS_TABLE_PATH = "agent_aliases.csv"

pd.options.display.float_format = "{:,.2f}".format

# =========================
# Agent alias utilities
# =========================

BASE36_CHARS = string.digits + string.ascii_uppercase

SPECIAL_ALIASES = {
    "AgeAgeAgent": "AGE",
    "SimpleAgent": "SIM",
    "OptimisticAgent": "OPT",
}


def to_base36(number: int) -> str:
    if number < 0:
        raise ValueError("number must be non-negative")

    if number == 0:
        return "0"

    digits = []
    while number:
        number, remainder = divmod(number, 36)
        digits.append(BASE36_CHARS[remainder])

    return "".join(reversed(digits))


def get_class_name(agent_cls: type) -> str:
    return getattr(agent_cls, "__name__", str(agent_cls).split(".")[-1])


def get_full_class_name(agent_cls: type) -> str:
    module_name = getattr(agent_cls, "__module__", "")
    class_name = get_class_name(agent_cls)
    return f"{module_name}.{class_name}" if module_name else class_name


def make_alias_prefix(class_name: str) -> str:
    if class_name in SPECIAL_ALIASES:
        return SPECIAL_ALIASES[class_name]

    cleaned = re.sub(r"Agent$", "", class_name)
    cleaned = re.sub(r"[^A-Za-z0-9]", "", cleaned)

    words = re.findall(
        r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+",
        cleaned,
    )

    initials = "".join(
        word[0].upper()
        for word in words
        if word and word[0].isalpha()
    )

    if len(initials) >= 2:
        return initials[:2]

    letters = re.sub(r"[^A-Za-z]", "", cleaned).upper()
    if len(letters) >= 2:
        return letters[:2]
    if len(letters) == 1:
        return letters + "X"

    return "AG"


def make_unique_alias(agent_cls: type, index: int, used_aliases: set[str]) -> str:
    class_name = get_class_name(agent_cls)
    prefix = make_alias_prefix(class_name)

    if len(prefix) == 3:
        alias = prefix
    else:
        suffix = to_base36(index)
        alias = f"{prefix[:2]}{suffix}"[:3]

    retry_count = 0
    while alias in used_aliases or not alias.isidentifier() or not alias[0].isalpha():
        suffix = to_base36(retry_count)
        alias = f"A{suffix.zfill(2)[-2:]}"
        retry_count += 1

    used_aliases.add(alias)
    return alias


def make_short_agent_class(base_cls: type, short_name: str) -> type:
    """
    既存エージェントクラスを継承して、短い名前のクラスを作る。

    重要:
        この関数はトップレベルで実行する。
        Windows multiprocessing の子プロセスでも同じ alias クラスを作るため。
    """
    short_cls = type(
        short_name,
        (base_cls,),
        {
            "__module__": __name__,
            "__doc__": f"Short alias of {get_full_class_name(base_cls)}.",
        },
    )

    globals()[short_name] = short_cls
    return short_cls


def build_aliased_competitors(agent_classes: list[type]) -> tuple[list[type], pd.DataFrame]:
    used_aliases: set[str] = set()
    aliased_agents = []
    rows = []

    for index, agent_cls in enumerate(agent_classes):
        alias = make_unique_alias(agent_cls, index, used_aliases)
        short_cls = make_short_agent_class(agent_cls, alias)

        aliased_agents.append(short_cls)
        rows.append(
            {
                "alias": alias,
                "original_class": get_class_name(agent_cls),
                "original_module": getattr(agent_cls, "__module__", ""),
                "original_full_name": get_full_class_name(agent_cls),
            }
        )

    alias_table = pd.DataFrame(rows)
    return aliased_agents, alias_table


# =========================
# Important:
# Create aliases at module top-level
# =========================

def load_original_tournament_types() -> list[type]:
    winners_2025 = get_agents(
        version=2025,
        track="std",
        winners_only=False,
        as_class=True,
    )

    winners_2025 = sorted(
        list(winners_2025),
        key=get_full_class_name,
    )

    rng = random.Random(RANDOM_SEED)
    sampled_winners = rng.sample(winners_2025, N_SAMPLED_WINNERS)

    original_agents = [AgeAgeAgent]

    original_agents += sampled_winners

    return original_agents


ORIGINAL_TOURNAMENT_TYPES = load_original_tournament_types()
TOURNAMENT_TYPES, ALIAS_TABLE = build_aliased_competitors(ORIGINAL_TOURNAMENT_TYPES)


# =========================
# Result utilities
# =========================

def shorten_type_name(value):
    if pd.isna(value):
        return value
    return str(value).split(".")[-1]


def add_original_agent_columns(results, alias_table: pd.DataFrame):
    alias_to_original = dict(
        zip(alias_table["alias"], alias_table["original_class"])
    )

    if hasattr(results, "score_stats"):
        results.score_stats.agent_type = results.score_stats.agent_type.map(shorten_type_name)
        results.score_stats["original_agent_type"] = (
            results.score_stats.agent_type.map(alias_to_original)
        )

    if hasattr(results, "total_scores"):
        results.total_scores.agent_type = results.total_scores.agent_type.map(shorten_type_name)
        results.total_scores["original_agent_type"] = (
            results.total_scores.agent_type.map(alias_to_original)
        )

    if hasattr(results, "scores"):
        results.scores.agent_type = results.scores.agent_type.map(shorten_type_name)
        results.scores["original_agent_type"] = (
            results.scores.agent_type.map(alias_to_original)
        )

    if hasattr(results, "kstest"):
        results.kstest.a = results.kstest.a.map(shorten_type_name)
        results.kstest.b = results.kstest.b.map(shorten_type_name)
        results.kstest["a_original"] = results.kstest.a.map(alias_to_original)
        results.kstest["b_original"] = results.kstest.b.map(alias_to_original)

    if hasattr(results, "winners"):
        results.winners = [shorten_type_name(winner) for winner in results.winners]

    return results


def plot_scores_by_level(results):
    if not hasattr(results, "scores"):
        print("results.scores が見つかりませんでした。")
        return

    scores = results.scores.copy()

    if "agent_name" not in scores.columns:
        print("agent_name 列が見つかりませんでした。")
        return

    scores["level"] = scores["agent_name"].astype(str).str.extract(r"@(\d+)")[0]
    scores = scores.dropna(subset=["level"])

    if scores.empty:
        print("level を抽出できませんでした。")
        return

    scores["level"] = scores["level"].astype(int)
    scores = scores.sort_values("level")

    sns.lineplot(
        data=scores[["agent_type", "level", "score"]],
        x="level",
        y="score",
        hue="agent_type",
    )

    plt.axhline(y=0.0, linestyle="--")
    plt.title("Score by Level")
    plt.tight_layout()
    plt.show()


# =========================
# Main
# =========================

if __name__ == "__main__":
    freeze_support()

    Path(TOURNAMENT_PATH).mkdir(parents=True, exist_ok=True)

    ALIAS_TABLE.to_csv(
        ALIAS_TABLE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== Agent aliases ===")
    print(ALIAS_TABLE[["alias", "original_class", "original_module"]].to_string(index=False))
    print(f"\nAlias table saved to: {ALIAS_TABLE_PATH}\n")

    results = anac2024_std(
        competitors=TOURNAMENT_TYPES,
        n_configs=5,
        n_competitors_per_world=len(TOURNAMENT_TYPES),
        n_runs_per_world=4,
        n_steps=125,
        print_exceptions=True,
        verbose=False,
        tournament_path=TOURNAMENT_PATH,
    )

    results = add_original_agent_columns(results, ALIAS_TABLE)

    print("\n=== Number of runs ===")
    print(len(results.scores.run_id.unique()))

    print("\n=== Score stats ===")
    print(results.score_stats)

    plot_scores_by_level(results)