from __future__ import annotations
import os

from scml.std import *
from scml.oneshot import *
from negmas import ResponseType
from scml_agents import get_agents
from typing import Any
import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio
import random
import time
from collections import defaultdict
from negmas import Contract, ResponseType, SAOResponse, SAOState
pio.renderers.default = "browser"
#!/usr/bin/env python

import random
from collections import defaultdict, deque

from scml.oneshot.common import QUANTITY, TIME, UNIT_PRICE

import random
from collections import Counter, defaultdict
from itertools import chain, combinations, repeat

# required for typing
from negmas import *
from numpy.random import choice

# required for development
from scml.std import *
from AgeAgeAgent import AgeAgeAgent

from pathlib import Path
from make_scml_log_viewer import generate_html_log
import webbrowser



class SimpleAgent(StdAgent):
    """A greedy agent based on StdAgent"""

    def __init__(self, *args, production_level=0.25, future_concession=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.production_level = production_level
        self.future_concession = future_concession

    def propose(self, negotiator_id: str, state):
        return self.good_offer(negotiator_id, state)

    def respond(self, negotiator_id, state, source=""):
        # accept any quantity needed at a good price
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
        # ending the negotiation is bad
        if offer is None:
            return False
        nmi = self.get_nmi(partner)
        if not nmi:
            return False
        issues = nmi.issues
        minp = issues[UNIT_PRICE].min_value
        maxp = issues[UNIT_PRICE].max_value
        # use relative negotiation time to concede
        # for offers about today but conede less for
        # future contracts
        r = state.relative_time
        if offer[TIME] > self.awi.current_step:
            r *= self.future_concession
        # concede linearly
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
            # find my needs for this day
            needed = self._needs(partner, t)
            if needed <= 0:
                continue
            offer = [-1] * 3
            # ask for as much as I need for this day
            offer[QUANTITY] = max(min(needed, qissue.max_value), qissue.min_value)
            offer[TIME] = t
            # use relative negotiation time to concede
            # for offers about today but conede less for
            # future contracts
            r = state.relative_time
            if t > self.awi.current_step:
                r *= self.future_concession
            # concede linearly on price
            minp, maxp = pissue.min_value, pissue.max_value
            if self.is_consumer(partner):
                offer[UNIT_PRICE] = int(minp + (maxp - minp) * (1 - r) + 0.5)
            else:
                offer[UNIT_PRICE] = int(minp + (maxp - minp) * r + 0.5)
            return tuple(offer)
        # just end the negotiation if I need nothing
        return None

    def is_consumer(self, partner):
        return partner in self.awi.my_consumers

    def _needs(self, partner, t):
        # find my needs today
        if self.awi.is_first_level:
            total_needs = self.awi.needed_sales
        elif self.awi.is_last_level:
            total_needs = self.awi.needed_supplies
        else:
            total_needs = self.production_level * self.awi.n_lines
        # estimate future needs
        if self.is_consumer(partner):
            total_needs += (
                self.production_level * self.awi.n_lines * (t - self.awi.current_step)
            )
            total_needs -= self.awi.total_sales_until(t)
        else:
            total_needs += (
                self.production_level * self.awi.n_lines * (self.awi.n_steps - t - 1)
            )
            total_needs -= self.awi.total_supplies_between(t, self.awi.n_steps - 1)
        # subtract already signed contracts
        return int(total_needs)


def distribute(q: int, n: int) -> list[int]:
    if n <= 0:
        return []
    if q <= 0:
        return [0] * n

    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst

    if q == n:
        return [1] * n

    r = Counter(choice(n, q - n))
    return [r.get(i, 0) + 1 for i in range(n)]


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    
def export_and_plot_stats(stats_df: pd.DataFrame, excel_path: str = "stats.xlsx", show=True) -> None:
    """
    world.stats_df を
    1. Excel に保存
    2. 10個のグラフを 1ウィンドウ(2x5) にまとめて表示
    """

    stats_df.to_excel(excel_path, index_label="step")

    if not show:
        return
    
    x = stats_df.index

    plot_specs = [
        ("trading_price_", "Trading Price", "price"),
        ("sold_quantity_", "Sold Quantity", "quantity"),
        ("unit_price_", "Unit Price", "price"),
        ("score_", "Score", "score"),
        ("balance_", "Balance", "balance"),
        ("productivity_", "Productivity", "productivity"),
        ("shortfall_penalty_", "Shortfall Penalty", "penalty"),
        ("inventory_penalized_", "Inventory Penalized", "quantity"),
        ("inventory_input_", "Inventory Input", "quantity"),
        ("inventory_output_", "Inventory Output", "quantity"),
    ]

    # グラフの描画設定
    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    axes = axes.flatten()

    for ax, (prefix, title, ylabel) in zip(axes, plot_specs):
        cols = [c for c in stats_df.columns if c.startswith(prefix)]

        if not cols:
            ax.set_title(f"{title}\n(no data)")
            ax.set_xlabel("step")
            ax.set_ylabel(ylabel)
            ax.grid(True)
            continue

        for col in sorted(cols):
            label = col[len(prefix):]
            ax.plot(x, stats_df[col], marker="o", linewidth=1.5, markersize=3, label=label)

        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.show()

def format_time(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def parquet_to_txt(file_names):
    """
    指定した複数のparquetファイルを読み込んで、
    カレントディレクトリに.txtで保存する関数
    """
    base_path = r"C:\Users\2kame\negmas\logs\test_world"

    # 表示省略なし設定
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

    for file_name in file_names:
        input_path = os.path.join(base_path, file_name)
        
        # 出力ファイル名（.parquet → .txt）
        output_name = os.path.splitext(file_name)[0] + ".txt"
        output_path = os.path.join("./data", output_name)

        try:
            df = pd.read_parquet(input_path)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(df.to_string())

            print(f"{file_name} → {output_name} 保存完了✨")

        except Exception as e:
            print(f"{file_name} でエラー: {e}")
            
if __name__ == '__main__':
    #エージェント取得
    all_agents_2024 = get_agents(version=2024, track="std", winners_only=False, as_class=True)
    all_agents_2025 = get_agents(version=2025, track="std", winners_only=False, as_class=True)
    print(all_agents_2024)
    name_map_2024 = {cls.__name__: cls for cls in all_agents_2024}
    name_map_2025 = {cls.__name__: cls for cls in all_agents_2025}

    #エージェントの担当工場を変更する場合、typesのエージェントの順番を変える
    types = [
        # AS0_log,
        AgeAgeAgent, 
        name_map_2025["AS0"],
        # name_map_2025["ProactiveAgent"], 
        name_map_2025["KATSUDONAgent"], 
        name_map_2025["PonponAgent"], 
        name_map_2025["XenoSotaAgent"], 
        name_map_2024["PenguinAgent"], 
        name_map_2024["AX"], 
        AgeAgeAgent, 
        name_map_2025["AS0"],
        # name_map_2025["OptimisticAgent"], 
        # name_map_2024["Group2"], 
        name_map_2024["AX"], 
        # name_map_2024["DogAgent"], 
        name_map_2024["MatchingPennies"], 
        name_map_2025["AS0"], 
        name_map_2024["CautiousStdAgent"], 
        AgeAgeAgent, 
        # name_map_2025["AS0"],
        # name_map_2024["QuickDecisionAgent"], 
    ]

    #シミュレーション設定
    world = SCML2024StdWorld(
        **SCML2024StdWorld.generate(
            agent_types = types,
            agent_processes=[0]*4 + [1]*5 + [2]*5,
            n_processes=3,
            n_steps=125,
            construct_graphs=True,
            random_agent_types=False,
            name="test_world",
            
            # n_competitors_per_world=len(types),
        )
    )
    world.init()
    total_time = 0.0

    #シミュレーション実行
    for step in range(world.n_steps):
        start = time.perf_counter()
        world.step()
        elapsed = time.perf_counter() - start
        total_time += elapsed

        eta = (total_time / (step + 1)) * (world.n_steps - step - 1)

        print(
            f"step {step + 1} / {world.n_steps}  |  "
            f"elapsed: {format_time(total_time)}  |  "
            f"ETA: {format_time(eta)}"
        )

    parquet_to_txt([
        "negs.parquet",
        "actions.parquet",
        "simsteps.parquet",
        "agents.parquet",
    ])
    #シミュレーション結果の出力
    # world.draw(steps=(0, world.n_steps-1), what=["negotiations-started", "contracts-concluded"], together=False, figsize=(50, 13))
    export_and_plot_stats(world.stats_df, "stats.xlsx", False)
    print("\n===== Time Summary =====")
    print(f"Total time: {total_time:.4f} sec")
    print(f"Avg per step: {total_time / world.n_steps:.4f} sec")

    generate_html_log()
    html_path = Path("scml_log_viewer.html").resolve()

    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

    webbrowser.register(
        "chrome",
        None,
        webbrowser.BackgroundBrowser(chrome_path)
    )

    webbrowser.get("chrome").open(html_path.as_uri())