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
        name_map_2025["KATSUDONAgent"], 
        # AS0_log,
        # name_map_2025["ProactiveAgent"], 
        name_map_2025["PriceTrendStdAgent"], 
        name_map_2025["AS0"],
        AgeAgeAgent, 
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
            n_steps=50,
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