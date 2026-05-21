#!/usr/bin/env python
# 旧

from __future__ import annotations

from itertools import repeat
import random
from collections import defaultdict
from typing import Literal
import math

from negmas import *
from scml.std import *

from BaseAgent import BaseAgent

from dataclasses import dataclass

__all__ = ["AgeAgeAgent"]

@dataclass
class TradeStats:
    success_count: int = 0
    fault_count: int = 0

class AgeAgeAgent(BaseAgent):
    # 何もしない
    NO_FIRST_PROPOSAL = False

    # 改善した機能のオンオフ
    BASE_AGENT_FIRST_PROPOSALS = False
    BASE_AGENT_COUNTER_ALL = False
    BASE_AGENT_DISTRIBUTION = False
    BETTER_COUNTER_ALL = True

    QUANTITY_AVG_DECAY = 0.7 # 取引量の加重平均の割引率
    AVG_DECREASE_ON_FAULT = 1 # 取引に失敗したときに加重平均をどれくらい減らすか

    partner_weighted_avg_quantity: dict[str, float]
    # 初回提案の内容を一時的に保持するための変数
    partner_first_offer: dict[str, tuple[int, int, int]] 
    quantity_adjust: dict[str, int]

    def __init__(self, *args, threshold=None, ptoday=0.70, productivity=0.7, **kwargs):
        super().__init__(*args, **kwargs)

        # experimental
        if not self.BASE_AGENT_DISTRIBUTION:
            self.history_table: dict[tuple[str, int, int, int], TradeStats] = defaultdict(TradeStats)
    
        # 加重平均の計算を、negotiationsuccess, negotiation failture, counter allで行う
        # ついでに交渉テーブルも作りたい
        self.partner_weighted_avg_quantity = defaultdict(float)
        self.partner_first_offer = {}
        self.quantity_adjust = defaultdict(int)
        
    def on_negotiation_success(self, contract, mechanism):
        if self.BASE_AGENT_DISTRIBUTION:
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
        # print(f"success \n{contract}\n")

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        # 契約が成立しなかった交渉相手の取引量の加重平均を減らす
        partner = next(p for p in partners if p != self.id)
        current_quantity = self.partner_weighted_avg_quantity[partner]
        self.partner_weighted_avg_quantity[partner] = max(
            1,
            current_quantity - self.AVG_DECREASE_ON_FAULT
        )

        # print(f"fialture {partners}: {state}")
        
    def step(self):
        super().step()
        # print(self.id, self.awi.total_sales, self.awi.total_supplies, self.get_needs())

    def first_proposals(self):
        if self.BASE_AGENT_FIRST_PROPOSALS:
            return super().first_proposals()
        if self.awi.current_step == 0:
            self.init_partner_avg_quantity(self.negotiators.keys())

        offers = {}
        buy_offers = {}
        sell_offers = {}
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
                buy_offers[partner] = offers[partner]
            else:
                sell_offers[partner] = offers[partner]

        # 納期を決定
        response |= self.assign_delivery_steps_by_knapsack(buy_offers, "buy_offer", self.awi.current_step, True)
        response |= self.assign_delivery_steps_by_knapsack(sell_offers, "sell_offer", self.awi.current_step, True)

        # print("生成したこちらからのオファー: ", offers)
        # print("エージェントごとの最適量: ", distribution)
        # print(self.id, ": ナップサックによって選ばれたオファー: ", response)
        
        # print(self.awi.current_step, self.get_needs())
        # print(self.get_needs())
        return response 

    def counter_all(self, offers, states):
        # print("counter offer\n", offers)

        if self.BASE_AGENT_COUNTER_ALL:
            return super().counter_all(offers, states)

        response = {}
        buy_offers = {}
        sell_offers = {}
        
        # 買い契約と売り契約に仕分け
        for partner, offer in offers.items():
            response[partner] = SAOResponse(
                ResponseType.END_NEGOTIATION, None
            )
            if partner in self.awi.my_suppliers:
                price_issue = self.awi.current_input_issues[UNIT_PRICE]
                buy_offers[partner] = offer
            else:
                sell_offers[partner] = offer

        # 最適なオファーの組み合わせを探索
        current_needs_supply, current_needs_consume = self.get_needs()
        _, selected_partners_supply = solve_knapsack_for_scml_offers(buy_offers, current_needs_supply)
        _, selected_partners_consume = solve_knapsack_for_scml_offers(sell_offers, current_needs_consume*2)

        # 受諾リストを作成
        for partner in selected_partners_supply:
            response[partner] = SAOResponse(
                ResponseType.ACCEPT_OFFER, None
            )
            
        for partner in selected_partners_consume:
            response[partner] = SAOResponse(
                ResponseType.ACCEPT_OFFER, None
            )

        if not self.BETTER_COUNTER_ALL:
            return response

        # 改善されたカウンターオール

        #==================
        # 試験的実装！！リファクタリング必須！！
        #=================
        response = {}
        # 相手から来たオファーに対しこちらの理想的な納期を設定
        offers = self.assign_delivery_steps_by_knapsack(buy_offers, "buy_offer", self.awi.current_step)

        for partner, offer in offers.items():
            # print(buy_offers[partner][TIME], offer[TIME], self.awi.current_step)
            if offer[TIME] == buy_offers[partner][TIME]:
                response[partner] = SAOResponse(
                    ResponseType.ACCEPT_OFFER, None
                )
            else:
                response[partner] = SAOResponse(
                    ResponseType.REJECT_OFFER, offer
                )
                    
        # 売りオファー
        offers = self.assign_delivery_steps_by_knapsack(sell_offers, "sell_offer", self.awi.current_step)

        for partner, offer in offers.items():
            # print(sell_offers[partner][TIME], offer[TIME], self.awi.current_step)
            if offer[TIME] == sell_offers[partner][TIME]:
                response[partner] = SAOResponse(
                    ResponseType.ACCEPT_OFFER, None
                )
            else:
                response[partner] = SAOResponse(
                    ResponseType.REJECT_OFFER, offer
                )
        # print("\nsupply needs: ", current_needs_supply, " consume needs: ", current_needs_consume)
        # print(selected_partners_consume, selected_partners_supply)
        # print("response: ", response)
        return response
    
    def distribute_todays_needs(self, partners=None) -> dict[str, int]:
        """
        Returns:
            エージェントIDをキー、取引量を値とする辞書
        """
        if partners is None:
            partners = self.negotiators.keys()

        if self.NO_FIRST_PROPOSAL:
            return dict(zip(partners, repeat(0)))

        if self.BASE_AGENT_DISTRIBUTION:
            return super().distribute_todays_needs()
        
        # 単純にこれまでの取引量の加重平均を取引量を返す
        response = {}
        for partner in partners:
            response[partner] = round(self.partner_weighted_avg_quantity[partner])
        return response

    def assign_delivery_steps_by_knapsack(self, offers, mode: str, step=0, is_first_proposals=False):
        """
        量と価格が決まっているオファーに対し、引数stepにおける必要量から動的計画法によって最適な納期を割り当てるメソッド
        Args:
            mode:
                buy_offer: 買いオファー
                sell_offer: 売りオファー
        Returns:
            offers
        """

        response = {}
        price_mode = "low" if mode == "buy_offer" else "high"
        remaining_offers = offers.copy()
        needs: int

        if mode == "buy_offer":
            needs, _ = self.get_needs(step, is_first_proposals)
        elif mode == "sell_offer":
            _, needs = self.get_needs(step, is_first_proposals)
        else:
            return response

        # 終了条件
        if step > self.awi.n_steps-1:
            return response
        
        # 動的計画法
        _, selected_partners = solve_knapsack_for_scml_offers(offers, needs, price_mode)

        for partner in selected_partners:
            response[partner] = (
                remaining_offers[partner][QUANTITY],
                step,
                remaining_offers[partner][UNIT_PRICE]
            )

            remaining_offers.pop(partner)

        # このstepで使わないオファーは次のstepで使う
        if len(remaining_offers) > 0:
            response |= self.assign_delivery_steps_by_knapsack(remaining_offers, mode, step+1)

        return response
    
    def get_needs(self, step=None, is_first_proposals=False):
        """
        当日の必要量を求めるメソッド
        Returns:
            buy_needs, sell_needs
        """
        awi = self.awi
        day_production = awi.n_lines * self._productivity
        if step==None:
            step=awi.current_step

        # 仕入れたい数(inventory input高すぎて基本負数)
        buy_needs = int(
            max(
                # 契約済み売り取引量 - 在庫 - 契約済み買い取引量 + 最大生産能力に対する不足分の50%
                0,
                awi.total_sales_at(step)
                - awi.current_inventory_input
                - awi.total_supplies_at(step)
                + (awi.n_lines - awi.total_sales_at(step))
            )
        )
        # 売りたい数(何か間違いがありそう)
        sell_needs = int(
            max(
                0,
                awi.n_lines
                - awi.total_sales_at(step),
            )
        )

        if is_first_proposals and step in range(awi.current_step, awi.current_step+3):
            buy_needs = buy_needs * 2
            sell_needs = int(sell_needs * 1.5)

        return buy_needs, sell_needs
        
    def _update_partner_avg_quantity(self, partner, quantity):
        """
        加重平均の計算
        """
        current_quantity = self.partner_weighted_avg_quantity[partner]
        next_quantity = quantity

        self.partner_weighted_avg_quantity[partner] = (
            (1-self.QUANTITY_AVG_DECAY) * current_quantity + self.QUANTITY_AVG_DECAY * next_quantity
        )

    def init_partner_avg_quantity(self, partners) -> None:
        """
        交渉パートナーの取引量の初期値をセット
        初期値は、必要量を人数で分割
        """
        buy_needs, sell_needs = self.get_needs(0, True)

        for partner in partners:
            self.partner_weighted_avg_quantity[partner] = (
                math.ceil(buy_needs / len(self.awi.my_suppliers))
                if self.is_supplier(partner)
                else math.ceil(sell_needs / len(self.awi.my_consumers))
            )

def solve_knapsack_for_scml_offers(
    offers: dict[str, tuple[int, int, int]],
    capacity: int,
    price_mode: Literal["high", "low"] = "high",
    max_unit_price: int | None = None,
) -> tuple[int, list[str]]:

    if capacity <= 0 or not offers:
        return 0, []

    if price_mode not in ("high", "low"):
        raise ValueError('price_mode must be "high" or "low"')

    partners = list(offers.keys())
    n = len(partners)

    if price_mode == "low" and max_unit_price is None:
        max_unit_price = max(offer[UNIT_PRICE] for offer in offers.values())

    def calc_value(offer: tuple[int, int, int]) -> int:
        quantity = offer[QUANTITY]
        unit_price = offer[UNIT_PRICE]

        if price_mode == "high":
            unit_value = unit_price
        else:
            # 安いほど価値が高い。
            # +1 しないと、全員同価格のとき価値0になって誰も選ばれない。
            unit_value = max_unit_price - unit_price + 1
            unit_value = max(1, unit_value)

        return quantity * unit_value

    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        partner = partners[i - 1]
        offer = offers[partner]

        quantity = offer[QUANTITY]
        value = calc_value(offer)

        for q in range(capacity + 1):
            dp[i][q] = dp[i - 1][q]

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