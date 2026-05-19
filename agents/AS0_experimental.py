#!/usr/bin/env python


from __future__ import annotations

from itertools import repeat
import random
from collections import defaultdict

# required for typing
from negmas import *

# required for development
from scml.std import *

from AS0 import AS0
__all__ = ["AS0"]

from dataclasses import dataclass

@dataclass
class TradeStats:
    success_count: int = 0
    fault_count: int = 0

class MyAS0(AS0):
    # 何もしない
    NO_FIRST_PROPOSAL = False

    # 改善した機能のオンオフ
    AS0_FIRST_PROPOSALS = False
    AS0_COUNTER_ALL = False
    AS0_DISTRIBUTION = False

    INITIAL_QUANTITY_RATIO = 0.1 # 0step目における提案取引量 maxp * ratio
    QUANTITY_AVG_DECAY = 0.7 # 取引量の加重平均
    FAULT_QUALITY_DECREASE = 1

    partner_weighted_avg_quantity: dict[str, float]
    # 初回提案の内容を一時的に保持するための変数
    partner_first_offer: dict[str, tuple[int, int, int]] 
    quantity_adjust: dict[str, int]

    def __init__(self, *args, threshold=None, ptoday=0.70, productivity=0.7, **kwargs):
        super().__init__(*args, **kwargs)

        # experimental
        if not self.AS0_DISTRIBUTION:
            self.history_table: dict[tuple[str, int, int, int], TradeStats] = defaultdict(TradeStats)
    
        # 加重平均の計算を、negotiationsuccess, negotiation failture, counter allで行う
        # ついでに交渉テーブルも作りたい
        self.partner_weighted_avg_quantity = defaultdict(float)
        self.partner_first_offer = {}
        self.quantity_adjust = defaultdict(int)
        
    def on_negotiation_success(self, contract, mechanism):
        if self.AS0_DISTRIBUTION:
            return 
        
        ##==============
        ## 改良した配分
        ##==============
        
        # 交渉結果テーブル作成

        partner = next(p for p in contract.partners if p != self.id)

        agreement = contract.agreement

        quantity = agreement["quantity"]
        delivery_time = agreement["time"]
        unit_price = agreement["unit_price"]

        self.history_table[
            partner,
            quantity,
            delivery_time - self.awi.current_step,
            unit_price,
        ].success_count += 1

        # 加重平均の計算
        self._update_partner_avg_quantity(partner, quantity)
        
        # print("avg quantitiy", partner, self.partner_weighted_avg_quantity[partner])
        print(f"success \n{contract}\n")

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        # 契約が成立しなかった交渉相手の取引量の加重平均を減らす
        partner = next(p for p in partners if p != self.id)
        current_quantity = self.partner_weighted_avg_quantity[partner]
        self.partner_weighted_avg_quantity[partner] = max(
            1,
            current_quantity - self.FAULT_QUALITY_DECREASE
        )

        # print(f"fialture {partners}: {state}")
        
    def step(self):
        super().step()

    def first_proposals(self):
        if self.AS0_FIRST_PROPOSALS:
            return super().first_proposals()
        if self.awi.current_step == 0:
            self.init_partner_avg_quantity(self.negotiators.keys())

        offers = {}
        supply_offers = {}
        consume_offers = {}
        response = {}

        # 取引量を決定
        distribution = self.distribute_todays_needs()

        # 価格を決定
        for partner, quantity in distribution.items():
            if quantity <= 0:
                continue

            price = self.smart_price(partner, is_first_proposal=True)

            if price is None:
                price_issue = self.awi.current_input_issues[UNIT_PRICE]
                if partner in self.awi.my_suppliers:
                    price = price_issue.min_value
                else:
                    price = price_issue.max_value

            offers[partner] = (
                quantity,
                self.awi.current_step,
                price,
            )

            if partner in self.awi.my_suppliers:
                price_issue = self.awi.current_input_issues[UNIT_PRICE]
                supply_offers[partner] = (
                    quantity,
                    self.awi.current_step,
                    max(1, price_issue.max_value - price) # 安いほうが利益が出るため、価値を逆転し、KP問題
                )
            else:
                consume_offers[partner] = offers[partner]

        # 動的計画法によって最適なオファーを選ぶ
        current_needs_supply, current_needs_consume = self.get_current_needs()

        _, selected_partners_supply = solve_knapsack_for_scml_offers(supply_offers, current_needs_supply)
        _, selected_partners_consume = solve_knapsack_for_scml_offers(consume_offers, current_needs_consume*2)

        # 選ばれたオファーだけの辞書を作成
        for partner, offer in offers.items():
            if not (partner in selected_partners_supply or partner in selected_partners_consume):
                continue
            response[partner] = offer

        print("supply needs: ", current_needs_supply, " consume needs: ", current_needs_consume)
        # print("生成したこちらからのオファー: ", offers)
        # print("エージェントごとの最適量: ", distribution)
        print("ナップサックによって選ばれたオファー: ", response)
        return response 

    def counter_all(self, offers, states):
        # print("counter offer\n", offers)

        if self.AS0_COUNTER_ALL:
            return super().counter_all(offers, states)

        response = {}
        supply_offers = {}
        consume_offers = {}
        
        # 買い契約と売り契約に仕分け
        for partner, offer in offers.items():
            response[partner] = SAOResponse(
                ResponseType.END_NEGOTIATION, None
            )
            if partner in self.awi.my_suppliers:
                price_issue = self.awi.current_input_issues[UNIT_PRICE]
                supply_offers[partner] = (
                    offer[QUANTITY],
                    offer[TIME],
                    max(1, price_issue.max_value - offer[UNIT_PRICE])# 安いほうが利益が出るので価値を逆転
                )
            else:
                consume_offers[partner] = offer

        # 最適なオファーの組み合わせを探索
        current_needs_supply, current_needs_consume = self.get_current_needs()
        _, selected_partners_supply = solve_knapsack_for_scml_offers(supply_offers, current_needs_supply)
        _, selected_partners_consume = solve_knapsack_for_scml_offers(consume_offers, current_needs_consume*2)

        # 受諾リストを作成
        for partner in selected_partners_supply:
            response[partner] = SAOResponse(
                ResponseType.ACCEPT_OFFER, None
            )
            
        for partner in selected_partners_consume:
            response[partner] = SAOResponse(
                ResponseType.ACCEPT_OFFER, None
            )

        # print("\nsupply needs: ", current_needs_supply, " consume needs: ", current_needs_consume)
        # print(selected_partners_consume, selected_partners_supply)
        print("response: ", response)
        return response
    
    def init_partner_avg_quantity(self, partners) -> None:
        """
        交渉パートナーの取引量の初期値をセット
        """
        for partner in partners:
            nmi = self.get_ami(partner)
            if nmi is None: continue
            
            quantity_issue = nmi.issues[QUANTITY]

            self.partner_weighted_avg_quantity[partner] = (
                quantity_issue.max_value + quantity_issue.min_value
            ) * self.INITIAL_QUANTITY_RATIO 

    def distribute_todays_needs(self, partners=None) -> dict[str, int]:
        """
        Returns:
            エージェントIDをキー、取引量を値とする辞書
        """
        if partners is None:
            partners = self.negotiators.keys()

        if self.NO_FIRST_PROPOSAL:
            return dict(zip(partners, repeat(0)))

        if self.AS0_DISTRIBUTION or self.AS0_FIRST_PROPOSALS:
            return super().distribute_todays_needs()
        
        # 単純にこれまでの取引量の加重平均を取引量とする
        response = {}
        for partner in partners:
            response[partner] = round(self.partner_weighted_avg_quantity[partner])
        return response
    
    def get_current_needs(self):
        """
        当日の必要量を求めるメソッド
        Return:
            supply_needs, consume_needs
        """
        awi = self.awi
        day_production = awi.n_lines * self._productivity
        # 仕入れたい数(inventory input高すぎて基本負数)
        supplie_needs = int(
            max(
                0,     
                day_production
                - awi.current_inventory_input
                - awi.total_supplies_at(awi.current_step)
            )
        )
        # 売りたい数(何か間違いがありそう)
        consume_needs = int(
            max(
                0,
                min(self.awi.n_lines, day_production + awi.current_inventory_input)
                - awi.total_sales_at(awi.current_step),
            )
        )

        return supplie_needs, consume_needs
        
    def _update_partner_avg_quantity(self, partner, quantity):
        """
        加重平均の計算
        """
        current_quantity = self.partner_weighted_avg_quantity[partner]
        next_quantity = quantity

        self.partner_weighted_avg_quantity[partner] = (
            (1-self.QUANTITY_AVG_DECAY) * current_quantity + self.QUANTITY_AVG_DECAY * next_quantity
        )

def solve_knapsack_for_scml_offers(
    offers: dict[str, tuple[int, int, int]],
    capacity: int,
) -> tuple[int, list[str]]:
    """
    SCMLのオファー集合から、数量制約内で価格合計が最大になる組み合わせを選ぶ。

    Args:
        offers:
            partner -> offer の辞書。
            offer は SCML の形式で (quantity, time, unit_price)

        capacity:
            受け入れ可能な最大数量。
            例: 今日必要な数量、awi.n_lines、在庫上限など。

    Returns:
        max_value:
            選んだオファーの合計価値。
            ここでは quantity * unit_price を価値とする。

        selected_partners:
            選ばれた partner のリスト。
    """

    partners = list(offers.keys())
    n = len(partners)

    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        partner = partners[i - 1]
        offer = offers[partner]

        quantity = offer[QUANTITY]
        unit_price = offer[UNIT_PRICE]

        value = quantity * unit_price

        for q in range(capacity + 1):
            # 選ばない場合
            dp[i][q] = dp[i - 1][q]

            # 選ぶ場合
            if quantity <= q:
                dp[i][q] = max(
                    dp[i][q],
                    dp[i - 1][q - quantity] + value,
                )

    selected_partners = []
    q = capacity

    for i in range(n, 0, -1):
        if dp[i][q] != dp[i - 1][q]:
            partner = partners[i - 1]
            selected_partners.append(partner)

            quantity = offers[partner][QUANTITY]
            q -= quantity

    selected_partners.reverse()

    return dp[n][capacity], selected_partners