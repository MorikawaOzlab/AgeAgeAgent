from __future__ import annotations

import os
import time
import webbrowser
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio
from negmas import ResponseType
from scml.oneshot.common import QUANTITY, TIME, UNIT_PRICE
from scml.std import *
from scml_agents import get_agents

from AgeAgeAgentV2 import AgeAgeAgentV2
from make_scml_log_viewer import generate_html_log

pio.renderers.default = "browser"


# =========================
# 表示・保存用ユーティリティ
# =========================

def export_and_plot_stats(
    stats_df: pd.DataFrame,
    excel_path: str = "stats.xlsx",
    show=True,
) -> None:
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
            ax.plot(
                x,
                stats_df[col],
                marker="o",
                linewidth=1.5,
                markersize=3,
                label=label,
            )

        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.show()


def format_time(sec: float) -> str:
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

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", None)

    os.makedirs("./data", exist_ok=True)

    for file_name in file_names:
        input_path = os.path.join(base_path, file_name)
        output_name = os.path.splitext(file_name)[0] + ".txt"
        output_path = os.path.join("./data", output_name)

        try:
            df = pd.read_parquet(input_path)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(df.to_string())

            print(f"{file_name} → {output_name} 保存完了✨")

        except Exception as e:
            print(f"{file_name} でエラー: {e}")


# =========================
# AgeAgeAgent の入出力監視用
# =========================

def _safe_offer_value(offer: Any, issue: int):
    if offer is None:
        return None

    try:
        return offer[issue]
    except Exception:
        return None


def format_offer(offer: Any) -> str:
    if offer is None:
        return "-"

    q = _safe_offer_value(offer, QUANTITY)
    t = _safe_offer_value(offer, TIME)
    p = _safe_offer_value(offer, UNIT_PRICE)

    if q is None or t is None or p is None:
        return repr(offer)

    return f"q={q}, t={t}, p={p}"


def response_name(response_type: Any) -> str:
    return getattr(response_type, "name", str(response_type))


def print_records(title: str, records: list[dict[str, Any]]) -> None:
    print(f"\n  [{title}]")

    if not records:
        print("    なし")
        return

    print("    partner                         side          round   action            offer")
    print("    " + "-" * 92)

    for r in records:
        partner = str(r.get("partner", ""))[:30]
        side = str(r.get("side", ""))[:12]
        round_no = r.get("round", "-")
        action = str(r.get("action", ""))[:16]
        offer = format_offer(r.get("offer"))

        print(
            f"    {partner:<30} "
            f"{side:<12} "
            f"{str(round_no):<7} "
            f"{action:<16} "
            f"{offer}"
        )


def print_ageage_logs_for_step(
    step: int,
    logs: dict[int, dict[str, dict[str, list[dict[str, Any]]]]],
) -> None:
    print("\n" + "=" * 110)
    print(f"AgeAgeAgent offer log | SCML step {step}")
    print("=" * 110)

    step_logs = logs.get(step, {})

    if not step_logs:
        print("この step では AgeAgeAgent のオファーログがありません。")
        return

    for agent_id, log in sorted(step_logs.items(), key=lambda x: str(x[0])):
        print(f"\n--- {agent_id} ---")
        print_records("過去に受諾済みの将来契約", log.get("accepted", []))
        print_records("来たオファー", log.get("incoming", []))
        print_records("価格チェックで落ちたオファー", log.get("price_rejected", []))
        print_records("出したオファー", log.get("outgoing", []))
        print_records("応答", log.get("responses", []))


def ask_continue() -> bool:
    """
    True なら次の step へ進む。
    False ならそこで終了する。
    """
    while True:
        cmd = input("\nEnter / y: 次のstepへ進む,  q: 終了 > ").strip().lower()

        if cmd in ("", "y", "yes", "next", "n"):
            return True

        if cmd in ("q", "quit", "exit", "stop"):
            return False

        print("Enter または y で続行、q で終了できます。")


class InspectableAgeAgeAgent(AgeAgeAgentV2):
    """
    AgeAgeAgent の挙動は変えず、
    first_proposals / counter_all で見えたオファーだけを記録するデバッグ用クラス。
    """

    step_logs: ClassVar[defaultdict] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "accepted": [],
                "incoming": [],
                "price_rejected": [],
                "outgoing": [],
                "responses": [],
            }
        )
    )

    def _my_side(self, partner: str) -> str:
        if partner in self.awi.my_suppliers:
            return "BUY/input"

        if partner in self.awi.my_consumers:
            return "SELL/output"

        return "UNKNOWN"

    def _record(
        self,
        kind: str,
        partner: str,
        offer: Any,
        action: str,
        round_no: Any = "-",
    ) -> None:
        self._record_for_step(
            step=self.awi.current_step,
            kind=kind,
            partner=partner,
            offer=offer,
            action=action,
            round_no=round_no,
        )

    def _record_for_step(
        self,
        step: int,
        kind: str,
        partner: str,
        offer: Any,
        action: str,
        round_no: Any = "-",
    ) -> None:
        """
        通常ログは current_step に記録する。
        ただし、過去に受諾した将来契約は、納期 step のログに先回りして記録する。
        """
        agent_id = self.id

        self.step_logs[step][agent_id][kind].append(
            {
                "partner": partner,
                "side": self._my_side(partner),
                "round": round_no,
                "action": action,
                "offer": offer,
            }
        )

    def first_proposals(self):
        proposals = super().first_proposals()

        for partner, offer in proposals.items():
            self._record(
                kind="outgoing",
                partner=partner,
                offer=offer,
                action="FIRST_PROPOSAL",
                round_no=0,
            )

        return proposals

    def counter_all(self, offers, states):
        valid_offers = {}
        price_rejected_offers = {}

        # 価格チェックを通ったものだけ「来たオファー」に表示する
        for partner, offer in offers.items():
            if self.is_min_profit_price(partner, offer[UNIT_PRICE]):
                valid_offers[partner] = offer
            else:
                price_rejected_offers[partner] = offer

        for partner, offer in valid_offers.items():
            state = states.get(partner)
            round_no = getattr(state, "step", "-") if state is not None else "-"

            self._record(
                kind="incoming",
                partner=partner,
                offer=offer,
                action="RECEIVED",
                round_no=round_no,
            )

        # 価格チェックで落ちたものは別枠で表示する
        for partner, offer in price_rejected_offers.items():
            state = states.get(partner)
            round_no = getattr(state, "step", "-") if state is not None else "-"

            self._record(
                kind="price_rejected",
                partner=partner,
                offer=offer,
                action="PRICE_REJECTED",
                round_no=round_no,
            )

        responses = super().counter_all(offers, states)

        # AgeAgeAgent が返した応答・カウンターオファー
        for partner, response in responses.items():
            state = states.get(partner)
            round_no = getattr(state, "step", "-") if state is not None else "-"

            response_type = getattr(response, "response", None)
            outcome = getattr(response, "outcome", None)
            action = response_name(response_type)

            display_offer = outcome

            # ACCEPT_OFFER の response.outcome は None になるため、
            # 受諾した内容は、そのラウンドで相手から来た offers[partner] を表示する。
            if response_type == ResponseType.ACCEPT_OFFER:
                display_offer = offers.get(partner)

            self._record(
                kind="responses",
                partner=partner,
                offer=display_offer,
                action=action,
                round_no=round_no,
            )

            # 過去に受諾した将来契約を、納期 step 側にも表示する。
            if response_type == ResponseType.ACCEPT_OFFER:
                delivery_step = _safe_offer_value(display_offer, TIME)
                accepted_at_step = self.awi.current_step

                if (
                    delivery_step is not None
                    and delivery_step > accepted_at_step
                    and delivery_step < self.awi.n_steps
                ):
                    self._record_for_step(
                        step=int(delivery_step),
                        kind="accepted",
                        partner=partner,
                        offer=display_offer,
                        action=f"ACCEPTED@{accepted_at_step}",
                        round_no=round_no,
                    )

            # REJECT_OFFER + outcome が、こちらが出したカウンターオファー
            if response_type == ResponseType.REJECT_OFFER and outcome is not None:
                self._record(
                    kind="outgoing",
                    partner=partner,
                    offer=outcome,
                    action="COUNTER_OFFER",
                    round_no=round_no,
                )

        return responses


# =========================
# メイン実行部
# =========================

if __name__ == "__main__":
    all_agents_2024 = get_agents(
        version=2024,
        track="std",
        winners_only=True,
        as_class=True,
    )

    all_agents_2025 = get_agents(
        version=2025,
        track="std",
        winners_only=True,
        as_class=True,
    )

    print(all_agents_2024)

    name_map_2024 = {cls.__name__: cls for cls in all_agents_2024}
    name_map_2025 = {cls.__name__: cls for cls in all_agents_2025}

    # AgeAgeAgent の代わりに InspectableAgeAgeAgent を使う。
    # 交渉ロジックは AgeAgeAgentV2 のまま、ログだけ追加される。
    types = [InspectableAgeAgeAgent] + list(all_agents_2024) + list(all_agents_2025)
    types = types + types

    world = SCML2024StdWorld(
        **SCML2024StdWorld.generate(
            agent_types=types,
            n_processes=3,
            n_steps=50,
            construct_graphs=True,
            random_agent_types=False,
            name="test_world",
        )
    )

    world.init()
    total_time = 0.0

    for step in range(world.n_steps):
        start = time.perf_counter()

        world.step()

        elapsed = time.perf_counter() - start
        total_time += elapsed

        eta = (total_time / (step + 1)) * (world.n_steps - step - 1)

        print(
            f"\nstep {step + 1} / {world.n_steps}  |  "
            f"elapsed: {format_time(total_time)}  |  "
            f"ETA: {format_time(eta)}"
        )

        print_ageage_logs_for_step(
            step,
            InspectableAgeAgeAgent.step_logs,
        )

        if step < world.n_steps - 1:
            if not ask_continue():
                print("\n手動停止しました。ここまでの結果を保存します。")
                break

    parquet_to_txt(
        [
            "negs.parquet",
            "actions.parquet",
            "simsteps.parquet",
            "agents.parquet",
        ]
    )

    export_and_plot_stats(world.stats_df, "stats.xlsx", False)

    print("\n===== Time Summary =====")
    print(f"Total time: {total_time:.4f} sec")
    print(f"Avg per executed step: {total_time / max(1, step + 1):.4f} sec")

    generate_html_log()
    html_path = Path("scml_log_viewer.html").resolve()

    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

    try:
        webbrowser.register(
            "chrome",
            None,
            webbrowser.BackgroundBrowser(chrome_path),
        )
        webbrowser.get("chrome").open(html_path.as_uri())
    except Exception as e:
        print(f"ChromeでHTMLログを開けませんでした: {e}")
        print(f"HTMLログ: {html_path}")