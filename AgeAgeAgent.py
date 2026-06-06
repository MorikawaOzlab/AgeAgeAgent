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

from dataclasses import dataclass, field
from typing import Any


__all__ = ["AgeAgeAgent"]

@dataclass
class OfferDecisionResult:
    accepted_responses: dict[str, SAOResponse] = field(default_factory=dict)
    counter_buy_offers: dict[str, Any] = field(default_factory=dict)
    counter_sell_offers: dict[str, Any] = field(default_factory=dict)

class AgeAgeAgent(StdSyncAgent):
    QUANTITY_AVG_DISCOUNT_RATE = 0.2 # 取引量の加重平均の割引率
    PRICE_AVG_DISCOUNT_RATE = 0.2
    AVG_DECREASE_ON_FAULT = 0.5 # 取引に失敗したときに加重平均をどれくらい減らすか

    NEAR_DELIVERY_WINDOW = 2
    
    MIN_PROFIT = -100

    avg_sell_price: float
    avg_buy_price: float

    partner_weighted_avg_quantity: dict[str, float]
    partner_weighted_avg_price: dict[str, float]
    # 初回提案の内容を一時的に保持するための変数
    partner_first_offer: dict[str, tuple[int, int, int]] 
    quantity_adjust: dict[str, int]
    partner_negotiation_counts: dict[str, tuple[int, int]]

    exo_input_q: int
    exo_output_q: int

    def __init__(self, *args, threshold=None, ptoday=0.70, productivity=0.7, **kwargs):
        super().__init__(*args, **kwargs)
    
        # 加重平均の計算を、negotiationsuccess, negotiation failture, counter allで行う
        # ついでに交渉テーブルも作りたい
        self.partner_weighted_avg_quantity = defaultdict(float)
        self.partner_weighted_avg_price = defaultdict(float)
        self.partner_first_offer = {}
        self.quantity_adjust = defaultdict(int)
        self.avg_buy_price = 0.0
        self.avg_sell_price = 0.0

    def step(self):
        awi = self.awi
        
        if awi.current_step == 0:
            partners = self.negotiators.keys()
            self.init_partner_avg_quantity(partners)
            self.init_partner_avg_price(partners)

        input_q = awi.current_exogenous_input_quantity
        output_q = awi.current_exogenous_output_quantity
        input_total_price = awi.current_exogenous_input_price
        output_total_price = awi.current_exogenous_output_price

        if input_q > 0:
            input_unit_price = input_total_price / input_q
            self.update_partner_avg_quantity("exogenous_input", input_q)
            self.update_partner_avg_price("exogenous_input", input_unit_price)
            return

        if output_q > 0:
            output_unit_price = output_total_price / output_q
            self.update_partner_avg_quantity("exogenous_output", output_q)
            self.update_partner_avg_price("exogenous_output", output_unit_price)
            return

    def on_negotiation_success(self, contract, mechanism):
        partner = next(p for p in contract.partners if p != self.id)

        agreement = contract.agreement

        quantity = agreement["quantity"]
        unit_price = agreement["unit_price"]

        # 加重平均の計算
        self.update_partner_avg_quantity(partner, quantity)

        # 平均取引価格の更新
        self.update_partner_avg_price(partner, unit_price)

    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        # 契約が成立しなかった交渉相手の取引量の加重平均を減らす
        partner = next(p for p in partners if p != self.id)
        current_quantity = self.partner_weighted_avg_quantity[partner]
        self.partner_weighted_avg_quantity[partner] = max(
            1,
            current_quantity - self.AVG_DECREASE_ON_FAULT
        )

    def first_proposals(self):
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

            if partner in self.awi.my_suppliers:
                offers[partner] = (
                    quantity,
                    self.awi.current_step,
                    self.get_valid_price(partner)
                )

                buy_offers[partner] = offers[partner]
            elif partner in self.awi.my_consumers:
                offers[partner] = (
                    quantity,
                    self.awi.current_step,
                    self.get_valid_price(partner)
                )

                sell_offers[partner] = offers[partner]

        # 納期を決定
        response |= self.assign_delivery_steps_by_knapsack(buy_offers, "buy_offer", self.awi.current_step, True)
        response |= self.assign_delivery_steps_by_knapsack(sell_offers, "sell_offer", self.awi.current_step, True)

        return response 

    def counter_all(self, offers, states):
        response = defaultdict(
            lambda: SAOResponse(ResponseType.END_NEGOTIATION, None)
        )

        buy_offers, sell_offers = (
            self.split_offers_by_partner(offers)
        )

        # # 価格のチェック
        # price_acceptable_buy_offers, price_adjusted_buy_offers = (
        #     self.check_offer_price(buy_offers, states)
        # )

        # price_acceptable_sell_offers, price_adjusted_sell_offers = (
        #     self.check_offer_price(sell_offers, states)
        # )

        # 納期ごとに必要な量の契約を結ぶ
        offer_decition_result = self.select_offers_by_delivery_step(buy_offers, sell_offers)

        response |= offer_decition_result.accepted_responses

        counter_buy_offers = offer_decition_result.counter_buy_offers
        counter_sell_offers = offer_decition_result.counter_sell_offers

        # 余ったオファーにこちらの理想的な納期を設定
        response |= self.make_counter_responses_by_knapsack(counter_buy_offers, "buy_offer", states)
        response |= self.make_counter_responses_by_knapsack(counter_sell_offers, "sell_offer", states)

        return response
    
    def distribute_todays_needs(self, partners=None) -> dict[str, int]:
        """
        Returns:
            エージェントIDをキー、取引量を値とする辞書
        """
        if partners is None:
            partners = self.negotiators.keys()
        
        response = {}
        total_seller_weight = 0
        total_buyer_weight = 0

        buy_needs, sell_needs = self.get_needs(None, True)

        # 全エージェントの平均取引量の合計を取得
        for partner in partners:
            if partner in self.awi.my_suppliers:
                total_seller_weight += self.partner_weighted_avg_quantity[partner]
            else:
                total_buyer_weight += self.partner_weighted_avg_quantity[partner]

        # 平均取引量で重みをつけて必要量を分配
        for partner in partners:
            if partner in self.awi.my_suppliers:
                if total_seller_weight == 0:
                    response[partner] = math.ceil(buy_needs / len(self.awi.my_suppliers))
                    continue

                response[partner] = math.ceil(
                    buy_needs * (self.partner_weighted_avg_quantity[partner] / total_seller_weight)
                )
            else:
                if total_buyer_weight == 0:
                    response[partner] = math.ceil(sell_needs / len(self.awi.my_consumers))
                    continue

                response[partner] = math.ceil(
                    sell_needs * (self.partner_weighted_avg_quantity[partner] / total_buyer_weight)
                )

        return response

    def assign_delivery_steps_by_knapsack(
            self, 
            offers, 
            mode: Literal["buy_offer", "sell_offer"],
            step=0, 
            is_first_proposals=False
        ):
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

        # 終了条件
        if step > self.awi.n_steps-1:
            return response

        if mode == "buy_offer":
            needs, _ = self.get_needs(step, is_first_proposals)
        elif mode == "sell_offer":
            _, needs = self.get_needs(step, is_first_proposals)
        else:
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
    
    def select_offers_by_delivery_step(self, buy_offers, sell_offers):
        result = OfferDecisionResult()

        # 納期ごとにオファーを分ける
        sorted_buy_offers = group_offers_by_delivery_time(buy_offers)
        sorted_sell_offers = group_offers_by_delivery_time(sell_offers)
        
        # 納期ごとにオファーの受諾判断
        for i in range(self.awi.current_step, self.awi.n_steps):
            buy_offer_dict = sorted_buy_offers.get(i, {})
            sell_offer_dict = sorted_sell_offers.get(i, {})

            #===========
            #価格チェック
            #===========

            offer_decition_result = self.select_offers_at_step(buy_offer_dict, sell_offer_dict, step=i)

            result.accepted_responses |= offer_decition_result.accepted_responses
            result.counter_buy_offers |= offer_decition_result.counter_buy_offers
            result.counter_sell_offers |= offer_decition_result.counter_sell_offers
            
        return result
    
    def select_offers_at_step(self, buy_offer_dict, sell_offer_dict, step):
        result = OfferDecisionResult()

        if len(buy_offer_dict) == 0 and len(sell_offer_dict) == 0:
            return result

        # 必要量計算
        target_buy_quantity, target_sell_quantity = (
            self.calculate_target_quantities_at_step(
                buy_offer_dict,
                sell_offer_dict,
                step,
            )
        )

        # ナップサック
        _, selected_supplier = solve_knapsack_for_scml_offers(
            buy_offer_dict,
            target_buy_quantity,
            "low",
        )

        _, selected_consumer = solve_knapsack_for_scml_offers(
            sell_offer_dict,
            target_sell_quantity,
            "high",
        )

        # 応答を作成
        for partner in selected_supplier + selected_consumer:
            result.accepted_responses[partner] = SAOResponse(
                ResponseType.ACCEPT_OFFER,
                None,
            )

        result.counter_buy_offers = buy_offer_dict.copy()
        for partner in selected_supplier:
            result.counter_buy_offers.pop(partner, None)

        result.counter_sell_offers = sell_offer_dict.copy()
        for partner in selected_consumer:
            result.counter_sell_offers.pop(partner, None)

        return result
    
    def calculate_target_quantities_at_step(
        self,
        buy_offer_dict,
        sell_offer_dict,
        step,
    ):
        """
        counter_all受諾判断に使う必要量計算
        Returns:
            target_buy_quantity, target_sell_quantity
        """
        total_buy_offer_quantity = get_total_offer_quantity(buy_offer_dict)
        total_sell_offer_quantity = get_total_offer_quantity(sell_offer_dict)

        contract_supply = self.awi.total_supplies_at(step)
        contract_sales = self.awi.total_sales_at(step)
        offer_supply = total_buy_offer_quantity
        offer_sales = total_sell_offer_quantity
        inventory = self.awi.current_inventory_input

        if self.awi.current_step <= step < self.awi.current_step + self.NEAR_DELIVERY_WINDOW:
            target_quantity = min(
                contract_sales + offer_sales,
                contract_supply + offer_supply + inventory,
                self.awi.n_lines,
            )

            input_q = self.awi.current_exogenous_input_quantity
            output_q = self.awi.current_exogenous_output_quantity

            if input_q > 0:
                target_quantity = min(
                    (inventory + input_q),
                    self.awi.n_lines,
                )

            elif output_q > 0:
                target_quantity = min(
                    max(
                        output_q,
                        int(
                            self.partner_weighted_avg_quantity["exogenous_output"]
                            * 1.1
                        ),
                    ),
                    self.awi.n_lines,
                )

            target_buy_quantity = max(
                0,
                target_quantity - contract_supply - inventory,
            )

            target_sell_quantity = max(
                0,
                target_quantity - contract_sales,
            )

        else:
            target_buy_quantity = max(
                0,
                contract_sales + offer_sales - contract_supply,
            )

            target_sell_quantity = max(
                0,
                self.awi.n_lines - contract_sales,
            )

        return target_buy_quantity, target_sell_quantity
       
    def make_counter_responses_by_knapsack(
            self, 
            counter_offers, 
            mode: Literal["buy_offer", "sell_offer"],
            states
        ):
        response = {}
        offers_new_delivery_steps = self.assign_delivery_steps_by_knapsack(counter_offers, mode, self.awi.current_step)

        for partner, offer in offers_new_delivery_steps.items():
            state = states.get(partner)
            new_offer = (
                offer[QUANTITY],
                offer[TIME],
                self.get_valid_price(partner, current_round=state.step)
            )
            response[partner] = SAOResponse(
                ResponseType.REJECT_OFFER, new_offer
            )

        return response

    def check_offer_price(self, offers, states):
        price_acceptable_offers = {}
        price_adjusted_offers = {}

        for partner, offer in offers.items():
            state = states.get(partner)

            # 適正価格よりも利益が出ない価格になっていた場合、修正してカウンターオファー
            if not self.is_valid_price(partner, offer[UNIT_PRICE]):
                new_offer = (
                    offer[QUANTITY],
                    offer[TIME],
                    self.get_valid_price(partner, current_round=state.step)
                )
                
                price_adjusted_offers[partner] = new_offer

                continue
            
            # 適正価格のオファーはそのまま返す
            price_acceptable_offers[partner] = offer

        return price_acceptable_offers, price_adjusted_offers
        
    def get_needs(self, step=None, is_first_proposals=False):
        """
        当日の必要量を求めるメソッド
        Returns:
            buy_needs, sell_needs
        """
        awi = self.awi
        if step==None:
            step=awi.current_step
        
        avg_sell_quantity = 0

        for partner, quantity in self.partner_weighted_avg_quantity.items():
            if partner in awi.my_consumers:
                avg_sell_quantity += quantity

        avg_sell_quantity = max(
            avg_sell_quantity / len(awi.my_consumers),
            self.partner_weighted_avg_quantity["exogenous_output"]
        )

        # 仕入れたい数(inventory input高すぎて基本負数)
        buy_needs = int(
            max(
                # 契約済み売り取引量 - 在庫 - 契約済み買い取引量 + 最大生産能力に対する不足分の50%
                0,
                awi.total_sales_at(step)
                - awi.current_inventory_input
                - awi.total_supplies_at(step)
                + (awi.n_lines - awi.total_sales_at(step)) * 0.7
            )
        )

        if is_first_proposals:
            buy_needs = int(buy_needs * 1.5)

        # 売りたい数(何か間違いがありそう)
        sell_needs = int(
            max(
                0,
                awi.n_lines
                - awi.total_sales_at(step),
            )
        )

        return buy_needs, sell_needs
        
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
    def update_partner_avg_quantity(self, partner, quantity):
        current_quantity = self.partner_weighted_avg_quantity[partner]
        next_quantity = quantity

        self.partner_weighted_avg_quantity[partner] = (
            (1-self.QUANTITY_AVG_DISCOUNT_RATE) * current_quantity + self.QUANTITY_AVG_DISCOUNT_RATE * next_quantity
        )

    def init_partner_avg_price(self, partners) -> None:
        """
        平均取引価格の初期値として、市場価格を設定
        """
        market_prices = self.awi.trading_prices
        
        input_market_price = market_prices[self.awi.my_input_product]
        output_market_price = market_prices[self.awi.my_output_product]

        for partner in partners:            
            if partner in self.awi.my_suppliers:
                self.partner_weighted_avg_price[partner] = input_market_price
                self.avg_buy_price = self.partner_weighted_avg_price[partner]
            else:
                self.partner_weighted_avg_price[partner] = output_market_price
                self.avg_sell_price = self.partner_weighted_avg_price[partner]

    
    def update_partner_avg_price(self, partner, price):
        """
        エージェントごとの平均取引価格と全取引の平均取引価格の更新
        """
        self.partner_weighted_avg_price[partner] = (
            (1 - self.PRICE_AVG_DISCOUNT_RATE)
            * self.partner_weighted_avg_price[partner]
            + self.PRICE_AVG_DISCOUNT_RATE
            * price
        )

        if partner in self.awi.my_suppliers or partner == "exogenous_input":
            self.avg_buy_price = (1 - self.PRICE_AVG_DISCOUNT_RATE) * self.avg_buy_price + self.PRICE_AVG_DISCOUNT_RATE * price
        elif partner in self.awi.my_consumers or  partner == "exogenous_output":
            self.avg_sell_price = (1 - self.PRICE_AVG_DISCOUNT_RATE) * self.avg_sell_price + self.PRICE_AVG_DISCOUNT_RATE * price
        
    def is_valid_price(self, partner, price):
        """
        オファーの価格が、十分利益の出るものになっているか判定する。
        """
        valid_price = self.get_valid_price(partner, 100)

        if partner in self.awi.my_suppliers:
            return price <= valid_price

        if partner in self.awi.my_consumers:
            return price >= valid_price

        return False
    
    # def get_valid_price(self, partner):
    #     price_issue = self.get_price_issue(partner)

    #     # 価格がMIN_PROFITの利益を確保できる値もしくはシステム上の上限or下限
    #     if partner in self.awi.my_suppliers:
    #         return max(price_issue.min_value, min(price_issue.max_value, int(self.avg_sell_price - self.MIN_PROFIT)))
    #     else:
    #         return min(price_issue.max_value, max(price_issue.min_value, int(self.avg_buy_price + self.MIN_PROFIT)))

    def get_valid_price(self, partner, current_round=0):
        price_issue = self.get_price_issue(partner)
        market_prices = self.awi.trading_prices

        input_market_price = market_prices[self.awi.my_input_product]
        output_market_price = market_prices[self.awi.my_output_product]

        if partner in self.awi.my_suppliers:
            initial_price = self.partner_weighted_avg_price[partner] * 0.9 # supplier min priceに変更
            max_price = int(math.ceil(input_market_price * 1.0))

            round_decay = (max_price - initial_price) / 3 * min(3, current_round)
            return max(price_issue.min_value, min(price_issue.max_value, math.ceil(initial_price + round_decay)))
        else:
            initial_price = self.partner_weighted_avg_price[partner] * 1.1
            min_price = output_market_price * 0.85
            round_decay = (initial_price - min_price) / 3 * min(3, current_round)
            return min(price_issue.max_value, max(price_issue.min_value, math.ceil(initial_price - round_decay)))

            # return min(price_issue.max_value, max(price_issue.min_value, int(output_market_price * 0.85)))
        
    def get_price_issue(self, partner):
        if partner in self.awi.my_suppliers:
            return self.awi.current_input_issues[UNIT_PRICE]
        else:
            return self.awi.current_output_issues[UNIT_PRICE]
    
    def split_offers_by_partner(self, offers):
        """
        Returns:
            buy_offers, sell_offers
        """
        buy_offers = {}
        sell_offers = {}

        for partner, offer in offers.items():
            if partner in self.awi.my_suppliers:
                buy_offers[partner] = offer

            elif partner in self.awi.my_consumers:
                sell_offers[partner] = offer
            
            else:
                continue

        return buy_offers, sell_offers
 
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

def group_offers_by_delivery_time(
    offers: dict[str, Outcome],
) -> dict[int, dict[str, Outcome]]:
    """
    オファーを納期ごとにグループ化する。

    Args:
        offers:
            エージェント名をキー、オファーを値に持つ辞書。
            例: {"agentA": (quantity, time, unit_price)}

    Returns:
        納期をキー、その納期のオファー集合を値に持つ辞書。
        例:
        {
            3: {"agentA": (5, 3, 20)},
            4: {"agentB": (2, 4, 18), "agentC": (1, 4, 19)}
        }
    """
    offers_by_time: dict[int, dict[str, Outcome]] = defaultdict(dict)

    for partner, offer in sorted(offers.items(), key=lambda item: item[1][TIME]):
        delivery_time = offer[TIME]
        offers_by_time[delivery_time][partner] = offer

    return dict(offers_by_time)

def get_total_offer_quantity(offers):
    """
    オファー集合から、取引量の合計値を返す
    """
    total_quantity = 0
    for _, offer in offers.items():
        total_quantity += offer[QUANTITY]
    
    return total_quantity