#!/usr/bin/env python
# 旧

from __future__ import annotations

from collections import defaultdict
from typing import Literal
import math

from negmas import *
from scml.std import *


from dataclasses import dataclass

__all__ = ["AgeAgeAgent"]

class AgeAgeAgent(StdSyncAgent):
    QUANTITY_AVG_DISCOUNT_RATE = 0.2 # 取引量の加重平均の割引率
    PRICE_AVG_DISCOUNT_RATE = 0.2
    AVG_DECREASE_ON_FAULT = 1 # 取引に失敗したときに加重平均をどれくらい減らすか

    MIN_PROFIT = 0

    avg_sell_price: float
    avg_buy_price: float
    buy_urgency_multiplier: float
    sell_urgency_multiplier: float

    partner_weighted_avg_quantity: dict[str, float]
    partner_weighted_avg_price: dict[str, float]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        # 加重平均の計算を、negotiationsuccess, negotiation failture, counter allで行う
        # ついでに交渉テーブルも作りたい
        self.partner_weighted_avg_quantity = defaultdict(float)
        self.partner_weighted_avg_price = defaultdict(float)
        self.avg_buy_price = 0.0
        self.avg_sell_price = 0.0
        self.buy_urgency_multiplier = 2.0
        self.sell_urgency_multiplier = 1.5
        
    def on_negotiation_success(self, contract, mechanism):
        # パートナーごとの平均値の更新
        partner = next(p for p in contract.partners if p != self.id)

        agreement = contract.agreement

        quantity = agreement["quantity"]
        delivery_time = agreement["time"]
        unit_price = agreement["unit_price"]

        self.update_partner_avg_quantity(partner, quantity)
        self.update_partner_avg_price(partner, unit_price)

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        # 契約が成立しなかった交渉相手の取引量の加重平均を減らす
        partner = next(p for p in partners if p != self.id)
        current_quantity = self.partner_weighted_avg_quantity[partner]
        self.partner_weighted_avg_quantity[partner] = max(
            1,
            current_quantity - self.AVG_DECREASE_ON_FAULT
        )

    def before_step(self):
        # 外生契約の価格平均、取引量平均の計算
        awi = self.awi
        input_q = awi.current_exogenous_input_quantity
        input_total_price = awi.current_exogenous_input_price

        if input_q <= 0:
            return
            
        input_unit_price = input_total_price / input_q
        self.update_partner_avg_price(None, input_unit_price, True)
        self.update_partner_avg_quantity("exogenous", input_q)

        if awi.current_step == 0:
            self.partner_weighted_avg_quantity["exogenous"] = input_q

    def first_proposals(self):
        current_step = self.awi.current_step

        if current_step == 0:
            partners = self.negotiators.keys()
            self.init_partner_avg_quantity(partners)
            self.init_partner_avg_price(partners)

        distribution = self.distribute_todays_needs()

        _, buy_offers, sell_offers = self.determine_price(distribution)

        proposals = self.assign_delivery_steps_by_knapsack(
            buy_offers,
            "buy_offer",
            current_step,
            True,
        )
        proposals |= self.assign_delivery_steps_by_knapsack(
            sell_offers,
            "sell_offer",
            current_step,
            True,
        )
        # print("ナップサックによって選ばれたオファー: ", proposals)
        return proposals 

    def counter_all(self, offers, states):
        if self.awi.my_input_product == 0:
            for partner, offer in offers.items():
                state = states[partner]

                round_index = state.step
                # print(round_index)

        response = {}
        buy_offers = {}
        sell_offers = {}
        
        # 利益が出るオファーかどうかチェック
        for partner, offer in offers.items():
            # 利益が出ないオファーは価格を修正してカウンターオファーを設定
            if not self.is_valid_price(partner, offer[UNIT_PRICE]):
                self.make_counter_offer_with_valid_price(response, offer, partner)
                continue

            buy_needs, sell_needs = self.get_needs(offer[TIME])

            # 利益が出るオファーは買いオファーと売りオファーに仕分ける
            response[partner] = SAOResponse(
                ResponseType.END_NEGOTIATION, None
            )
            self.add_offer_by_partner_type(partner, offer, buy_offers, sell_offers)

        # 納期がこちらの希望とあってるかチェック
        new_offers = self.assign_delivery_steps_by_knapsack(buy_offers, "buy_offer", self.awi.current_step)

        for partner, offer in new_offers.items():
            if offer[TIME] == buy_offers[partner][TIME]:
                response[partner] = SAOResponse(
                    ResponseType.ACCEPT_OFFER, None
                )
            else:
                new_offer = (
                    offer[QUANTITY],
                    offer[TIME],
                    self.get_valid_price(partner)
                )
                response[partner] = SAOResponse(
                    ResponseType.REJECT_OFFER, new_offer
                )
                    
        # 売りオファー
        new_offers = self.assign_delivery_steps_by_knapsack(sell_offers, "sell_offer", self.awi.current_step)

        for partner, offer in new_offers.items():
            if offer[TIME] == sell_offers[partner][TIME]:
                response[partner] = SAOResponse(
                    ResponseType.ACCEPT_OFFER, None
                )
            else:
                new_offer = (
                    offer[QUANTITY],
                    offer[TIME],
                    self.get_valid_price(partner)
                )
                response[partner] = SAOResponse(
                    ResponseType.REJECT_OFFER, new_offer
                )
        return response
    
    def distribute_todays_needs(self, partners=None) -> dict[str, int]:
        """
        Returns:
            エージェントIDをキー、取引量を値とする辞書
        """
        if partners is None:
            partners = self.negotiators.keys()
        
        # 単純にこれまでの取引量の加重平均を取引量を返す
        response = {}
        for partner in partners:
            response[partner] = round(self.partner_weighted_avg_quantity[partner])
        return response

    def determine_price(self, distribution):
        """
        取引量のみが設定されたオファーに利益の出る価格を設定して返す
        Args:
            distribution: 取引相手ごとの取引量。
        Returns:
            全オファー、買いオファー、売りオファーのタプル。
        """
        offers = {}
        buy_offers = {}
        sell_offers = {}

        for partner, quantity in distribution.items():
            if quantity <= 0:
                continue

            offers[partner] = (
                quantity,
                self.awi.current_step,
                self.get_valid_price(partner)
            )

            self.add_offer_by_partner_type(partner, offers[partner], buy_offers, sell_offers)
        
        return offers, buy_offers, sell_offers
    
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

        # 終了条件
        if step > self.awi.n_steps - 1:
            return response
        
        remaining_offers = offers.copy()
        needs: int

        if mode == "buy_offer":
            needs, _ = self.get_needs(step, is_first_proposals)
        elif mode == "sell_offer":
            _, needs = self.get_needs(step, is_first_proposals)
        else:
            return response
        
        # 動的計画法
        price_mode = "low" if mode == "buy_offer" else "high"
        _, selected_partners = solve_knapsack_for_scml_offers(offers, needs, price_mode)

        for partner in selected_partners:
            response[partner] = (
                remaining_offers[partner][QUANTITY],
                step,
                remaining_offers[partner][UNIT_PRICE]
            )

            remaining_offers.pop(partner)

        # このstepで使わないオファーは次のstepで使う
        if remaining_offers:
            response |= self.assign_delivery_steps_by_knapsack(remaining_offers, mode, step + 1, is_first_proposals)

        return response
    
    def get_needs(self, step=None, is_first_proposals=False):
        """
        当日の必要量を求めるメソッド
        Returns:
            buy_needs, sell_needs
        """
        awi = self.awi
        if step is None:
            step=awi.current_step

        if self.partner_weighted_avg_quantity["exogenous"] != 0:
            return self.urgency_multiplier(
                step, 
                is_first_proposals, 
                *self.get_exogenous_needs(step, is_first_proposals)
            )

        # 仕入れたい数(inventory input高すぎて基本負数)
        buy_needs = int(
            max(
                # 契約済み売り取引量 - 在庫 - 契約済み買い取引量 + 最大生産能力に対する不足分の70%
                0,
                awi.total_sales_at(step)
                - awi.current_inventory_input
                - awi.total_supplies_at(step)
                + (awi.n_lines - awi.total_sales_at(step)) * 0.7
            )
        )
        # 売りたい数
        sell_needs = int(
            max(
                0,
                awi.n_lines
                - awi.total_sales_at(step),
            )
        )

        return self.urgency_multiplier(step, is_first_proposals, buy_needs, sell_needs)
        
    def get_exogenous_needs(self, step, is_first_proposals):
        awi = self.awi
        exo_input = awi.current_exogenous_input_quantity
        exo_output = awi.current_exogenous_output_quantity
        
        buy_needs = sell_needs = 0
        if step == awi.current_step:
            buy_needs = exo_input if awi.is_last_level else 0
            sell_needs = exo_output if awi.is_first_level else 0
        else:
            buy_needs = math.ceil(self.partner_weighted_avg_quantity["exogenous"]) if awi.is_last_level else 0
            sell_needs = math.ceil(self.partner_weighted_avg_quantity["exogenous"]) if awi.is_first_level else 0
        
        return self.urgency_multiplier(step, is_first_proposals, buy_needs, sell_needs)
    
    def urgency_multiplier(
            self, 
            step, 
            is_first_proposals, 
            buy_needs, 
            sell_needs
    ):
        """
        緊急時に必要量を増やし、それ以外ではそのまま必要量を返す
        Returns:
            buy_needs, sell_needs
        """
        awi = self.awi
        if is_first_proposals and (awi.current_step <= step < awi.current_step + 3):
            buy_needs = int(buy_needs * self.buy_urgency_multiplier)
            sell_needs = int(sell_needs * self.sell_urgency_multiplier)
        
        return buy_needs, sell_needs

    def update_partner_avg_quantity(self, partner, quantity):
        """
        加重平均の計算
        """
        
        current_quantity = self.partner_weighted_avg_quantity[partner]
        next_quantity = quantity

        self.partner_weighted_avg_quantity[partner] = (
            (1-self.QUANTITY_AVG_DISCOUNT_RATE) * current_quantity + self.QUANTITY_AVG_DISCOUNT_RATE * next_quantity
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
                if partner in self.awi.my_suppliers
                else math.ceil(sell_needs / len(self.awi.my_consumers))
            )
    
    def update_partner_avg_price(self, partner, price, is_exogenous = False):
        self.partner_weighted_avg_price[partner] = (1 - self.PRICE_AVG_DISCOUNT_RATE) * self.partner_weighted_avg_price[partner] + self.PRICE_AVG_DISCOUNT_RATE * price

        if partner in self.awi.my_suppliers or is_exogenous:
            self.avg_buy_price = (1 - self.PRICE_AVG_DISCOUNT_RATE) * self.avg_buy_price + self.PRICE_AVG_DISCOUNT_RATE * price
        elif partner in self.awi.my_consumers or is_exogenous:
            self.avg_sell_price = (1 - self.PRICE_AVG_DISCOUNT_RATE) * self.avg_sell_price + self.PRICE_AVG_DISCOUNT_RATE * price
        
    def init_partner_avg_price(self, partners) -> None:
        for partner in partners:
            price_issue = self.get_price_issue(partner)
            
            if partner in self.awi.my_suppliers:
                self.partner_weighted_avg_price[partner] = price_issue.min_value
                self.avg_buy_price = self.partner_weighted_avg_price[partner]
            else:
                self.partner_weighted_avg_price[partner] = price_issue.max_value        
                self.avg_sell_price = self.partner_weighted_avg_price[partner]

    def is_valid_price(self, partner, price):
        """
        オファーの価格が、十分利益の出るものになっているか判定する。
        """
        valid_price = self.get_valid_price(partner)

        if partner in self.awi.my_suppliers:
            return price <= valid_price

        if partner in self.awi.my_consumers:
            return price >= valid_price

        return False
    
    def get_valid_price(self, partner):
        price_issue = self.get_price_issue(partner)

        # 価格がMIN_PROFITの利益を確保できる値もしくはシステム上の上限or下限
        if partner in self.awi.my_suppliers:
            return max(price_issue.min_value, min(price_issue.max_value, math.ceil(self.avg_sell_price - self.MIN_PROFIT)))
        else:
            return min(price_issue.max_value, max(price_issue.min_value, math.ceil(self.avg_buy_price + self.MIN_PROFIT)))
        
    def get_price_issue(self, partner):
        if partner in self.awi.my_suppliers:
            return self.awi.current_input_issues[UNIT_PRICE]
        else:
            return self.awi.current_output_issues[UNIT_PRICE]
    
    def add_offer_by_partner_type(self, partner, offer, buy_offers, sell_offers):
        """オファーを仕入れ先/販売先ごとに仕分ける。"""
        if partner in self.awi.my_suppliers:
            buy_offers[partner] = offer
        elif partner in self.awi.my_consumers:
            sell_offers[partner] = offer

    def make_counter_offer_with_valid_price(self, response, offer, partner):
        new_offer = (
            offer[QUANTITY],
            offer[TIME],
            self.get_valid_price(partner)
        )
        response[partner] = SAOResponse(
            ResponseType.REJECT_OFFER, new_offer
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