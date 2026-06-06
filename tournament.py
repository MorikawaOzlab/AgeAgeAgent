from scml.std import *
from negmas import ResponseType, SAOResponse
from scml.utils import anac2024_std

from scml_agents import get_agents

from collections import defaultdict, Counter
from pathlib import Path
import importlib
import random
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from AgeAgeAgent import AgeAgeAgent


class SimpleAgent(StdAgent):
    """A greedy agent based on StdAgent"""

    def __init__(self, *args, production_level=0.25, future_concession=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.production_level = production_level
        self.future_concession = future_concession

    def propose(self, negotiator_id: str, state):
        return self.good_offer(negotiator_id, state)

    def respond(self, negotiator_id, state, source=""):
        offer = state.current_offer
        return (
            ResponseType.ACCEPT_OFFER
            if self.is_needed(negotiator_id, offer)
            and self.is_good_price(negotiator_id, offer, state)
            else ResponseType.REJECT_OFFER
        )

    def is_needed(self, partner, offer):
        if offer is None:
            return False
        return offer[QUANTITY] <= self._needs(partner, offer[TIME])

    def is_good_price(self, partner, offer, state):
        if offer is None:
            return False

        nmi = self.get_nmi(partner)
        if not nmi:
            return False

        issues = nmi.issues
        minp = issues[UNIT_PRICE].min_value
        maxp = issues[UNIT_PRICE].max_value

        r = state.relative_time

        if offer[TIME] > self.awi.current_step:
            r *= self.future_concession

        if self.is_consumer(partner):
            return offer[UNIT_PRICE] >= minp + (1 - r) * (maxp - minp)

        return -offer[UNIT_PRICE] >= -minp + (1 - r) * (minp - maxp)

    def good_offer(self, partner, state):
        nmi = self.get_nmi(partner)
        if not nmi:
            return None

        issues = nmi.issues
        qissue = issues[QUANTITY]
        pissue = issues[UNIT_PRICE]

        for t in sorted(list(issues[TIME].all)):
            needed = self._needs(partner, t)

            if needed <= 0:
                continue

            offer = [-1] * 3

            offer[QUANTITY] = max(
                min(needed, qissue.max_value),
                qissue.min_value,
            )
            offer[TIME] = t

            r = state.relative_time

            if t > self.awi.current_step:
                r *= self.future_concession

            minp = pissue.min_value
            maxp = pissue.max_value

            if self.is_consumer(partner):
                offer[UNIT_PRICE] = int(minp + (maxp - minp) * (1 - r) + 0.5)
            else:
                offer[UNIT_PRICE] = int(minp + (maxp - minp) * r + 0.5)

            return tuple(offer)

        return None

    def is_consumer(self, partner):
        return partner in self.awi.my_consumers

    def _needs(self, partner, t):
        if self.awi.is_first_level:
            total_needs = self.awi.needed_sales
        elif self.awi.is_last_level:
            total_needs = self.awi.needed_supplies
        else:
            total_needs = self.production_level * self.awi.n_lines

        if self.is_consumer(partner):
            total_needs += (
                self.production_level
                * self.awi.n_lines
                * (t - self.awi.current_step)
            )
            total_needs -= self.awi.total_sales_until(t)
        else:
            total_needs += (
                self.production_level
                * self.awi.n_lines
                * (self.awi.n_steps - t - 1)
            )
            total_needs -= self.awi.total_supplies_between(
                t,
                self.awi.n_steps - 1,
            )

        return int(total_needs)


class OptimisticAgent(SimpleAgent):
    """A greedy agent based on SimpleAgent with more sane strategy"""

    def propose(self, negotiator_id, state):
        offer = self.good_offer(negotiator_id, state)

        if offer is None:
            return offer

        offered = self._offered(negotiator_id)
        offered[negotiator_id] = {offer[TIME]: offer[QUANTITY]}

        return offer

    def before_step(self):
        self.offered_sales = defaultdict(lambda: defaultdict(int))
        self.offered_supplies = defaultdict(lambda: defaultdict(int))

    def on_negotiation_success(self, contract, mechanism):
        partner = [_ for _ in contract.partners if _ != self.id][0]
        offered = self._offered(partner)
        offered[partner] = dict()

    def _offered(self, partner):
        if self.is_consumer(partner):
            return self.offered_sales

        return self.offered_supplies

    def _needs(self, partner, t):
        n = super()._needs(partner, t)
        offered = self._offered(partner)

        for k, v in offered[partner].items():
            if k > t:
                continue

            n = max(0, n - v)

        return int(n)


def make_short_name(old_name, duplicated_head3):
    """
    エージェント名を短縮する。

    基本:
        頭3文字

    頭3文字が他のエージェントと被る場合:
        頭1文字 + 尻2文字
    """
    if not old_name:
        return "Ag"

    if not duplicated_head3:
        return old_name[:3]

    if len(old_name) >= 3:
        return old_name[0] + old_name[-2:]

    return old_name


def make_short_names(agent_classes):
    """
    エージェントクラス一覧から短縮名を作る。

    例:
        AgeAgeAgent -> Age
        PriceTrendStdAgent -> Pri

    頭3文字が被る場合:
        LitaAgentN  -> LtN
        LitaAgentYS -> LYS

    それでも被る場合:
        LtN, LtN2, LtN3 ...
    """
    original_names = [cls.__name__ for cls in agent_classes]
    head3_counts = Counter(name[:3] for name in original_names)
    used_short_names = defaultdict(int)

    short_names = []

    for old_name in original_names:
        head3 = old_name[:3]
        duplicated_head3 = head3_counts[head3] >= 2

        base_name = make_short_name(old_name, duplicated_head3)

        used_short_names[base_name] += 1

        if used_short_names[base_name] == 1:
            short_name = base_name
        else:
            short_name = f"{base_name}{used_short_names[base_name]}"

        short_names.append(short_name)

    return short_names


def get_importable_module_name(cls):
    """
    クラスをimportできるモジュール名を返す。

    自分のファイル内で定義したクラスは __main__ になることがあるので、
    その場合はこのファイル名をモジュール名として使う。
    """
    if cls.__module__ == "__main__":
        return Path(__file__).stem

    return cls.__module__


def create_short_agent_alias_module(agent_classes, module_name="short_agent_aliases"):
    """
    短縮名のラッパークラスを定義したPythonファイルを自動生成する。

    重要:
        直接 cls.__name__ を変更しない。
        変更すると、SCML/negmas が元モジュールから短縮名クラスを探して失敗する。

    生成例:
        from AgeAgeAgent import AgeAgeAgent as _BaseAgent0

        class Age(_BaseAgent0):
            pass
    """
    short_names = make_short_names(agent_classes)

    module_path = Path(__file__).with_name(f"{module_name}.py")

    lines = []
    lines.append("# Auto-generated by tournament.py")
    lines.append("# This file defines short-name wrapper agents.")
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("")

    for i, (cls, short_name) in enumerate(zip(agent_classes, short_names)):
        import_module = get_importable_module_name(cls)
        original_name = cls.__name__

        lines.append(
            f"from {import_module} import {original_name} as _BaseAgent{i}"
        )
        lines.append("")
        lines.append(f"class {short_name}(_BaseAgent{i}):")
        lines.append("    pass")
        lines.append("")

    lines.append("__all__ = [")
    for short_name in short_names:
        lines.append(f'    "{short_name}",')
    lines.append("]")
    lines.append("")

    module_path.write_text("\n".join(lines), encoding="utf-8")

    importlib.invalidate_caches()

    if module_name in sys.modules:
        del sys.modules[module_name]

    alias_module = importlib.import_module(module_name)

    aliased_classes = [getattr(alias_module, short_name) for short_name in short_names]

    name_map = [
        (cls.__name__, short_name)
        for cls, short_name in zip(agent_classes, short_names)
    ]

    return aliased_classes, name_map


def unique_agent_classes(agent_classes):
    """
    同じクラスが重複していたら除外する。
    """
    unique = []
    seen = set()

    for cls in agent_classes:
        key = (cls.__module__, cls.__name__)

        if key in seen:
            continue

        seen.add(key)
        unique.append(cls)

    return unique


def shorten_names(results):
    """
    結果表示用に agent_type を短くする。
    """
    if hasattr(results, "score_stats") and "agent_type" in results.score_stats.columns:
        results.score_stats["agent_type"] = (
            results.score_stats["agent_type"].astype(str).str.split(".").str[-1]
        )

    if hasattr(results, "kstest"):
        if "a" in results.kstest.columns:
            results.kstest["a"] = (
                results.kstest["a"].astype(str).str.split(".").str[-1]
            )
        if "b" in results.kstest.columns:
            results.kstest["b"] = (
                results.kstest["b"].astype(str).str.split(".").str[-1]
            )

    if hasattr(results, "total_scores") and "agent_type" in results.total_scores.columns:
        results.total_scores["agent_type"] = (
            results.total_scores["agent_type"].astype(str).str.split(".").str[-1]
        )

    if hasattr(results, "scores") and "agent_type" in results.scores.columns:
        results.scores["agent_type"] = (
            results.scores["agent_type"].astype(str).str.split(".").str[-1]
        )

    if hasattr(results, "winners"):
        results.winners = [_.split(".")[-1] for _ in results.winners]

    return results


if __name__ == "__main__":
    pd.options.display.float_format = "{:,.2f}".format

    winners_2025 = get_agents(
        version=2025,
        track="std",
        winners_only=False,
        as_class=True,
    )

    winners_2024 = get_agents(
        version=2024,
        track="std",
        winners_only=False,
        as_class=True,
    )

    candidate_agents = list(winners_2025) + list(winners_2024)
    candidate_agents = unique_agent_classes(candidate_agents)

    tournament_types = [AgeAgeAgent] + random.sample(candidate_agents, 13) # max 18

    tournament_types, name_map = create_short_agent_alias_module(tournament_types)

    print("========== Selected Agents ==========")
    for old_name, short_name in name_map:
        print(f"{old_name} -> {short_name}")

    print()
    print("num agents:", len(tournament_types))
    print()

    results = anac2024_std(
        competitors=tournament_types,
        # 途中で止まる(57%)
        n_configs=5,
        n_competitors_per_world=len(tournament_types),
        n_runs_per_world=5,
        n_steps=50,
        print_exceptions=True,
        verbose=False,
        tournament_path=r"C:\t",
    )

    results = shorten_names(results)

    print("========== Number of Runs ==========")

    if hasattr(results, "scores") and "run_id" in results.scores.columns:
        print(len(results.scores.run_id.unique()))
    else:
        print("scores が取得できませんでした")

    print()

    print("========== Score Stats ==========")
    print(results.score_stats)
    print()

    if (
        hasattr(results, "scores")
        and "agent_name" in results.scores.columns
        and "agent_type" in results.scores.columns
        and "score" in results.scores.columns
    ):
        results.scores["level"] = (
            results.scores.agent_name.str.split("@", expand=True).loc[:, 1]
        )

        results.scores = results.scores.sort_values("level")

        sns.lineplot(
            data=results.scores[["agent_type", "level", "score"]],
            x="level",
            y="score",
            hue="agent_type",
        )

        plt.axhline(0.0, linestyle="--")
        plt.show()
    else:
        print("グラフ描画に必要な列がありません")