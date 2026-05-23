from scml.std import *
from negmas import ResponseType, SAOResponse
from scml.utils import anac2024_std

from scml_agents import get_agents

from collections import defaultdict
from itertools import repeat
import random
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
    
pd.options.display.float_format = '{:,.2f}'.format

def shorten_names(results):
    # just make agent types more readable
    results.score_stats.agent_type = results.score_stats.agent_type.str.split(".").str[-1]
    results.kstest.a = results.kstest.a.str.split(".").str[-1]
    results.kstest.b = results.kstest.b.str.split(".").str[-1]
    results.total_scores.agent_type = results.total_scores.agent_type.str.split(".").str[-1]
    results.scores.agent_type = results.scores.agent_type.str.split(".").str[-1]
    results.winners = [_.split(".")[-1] for _ in results.winners]
    return results

# 昨年の優勝エージェントを取得
# 他の年のエージェントを入れる場合はversionを変更
winners_2025 = get_agents(version=2025, track="std", winners_only=False, as_class=True)

# tournament_types = [SimpleAgent, OptimisticAgent, AgeAgeAgent] + winners_2025 #random.sample(list(winners_2025), 8) #自分のエージェントクラスをここに追加して実行
tournament_types = [AgeAgeAgent] + random.sample(list(winners_2025), 7)

if __name__ == '__main__':
    results = anac2024_std(
        competitors=tournament_types,
        n_configs=1, # number of different configurations to generate
        n_competitors_per_world=len(tournament_types),
        n_runs_per_world=2, # number of times to repeat every simulation (with agent assignment)
        n_steps=50, # number of days (simulation steps) per simulation 本番は50, 125, 200
        print_exceptions=False,
        verbose = True,
        tournament_path=r"C:\tmp\scml_t",
    )

    results = shorten_names(results)

    print(len(results.scores.run_id.unique()))

    print(results.score_stats)

    results.scores["level"] = results.scores.agent_name.str.split("@", expand=True).loc[:, 1]
    results.scores = results.scores.sort_values("level")
    sns.lineplot(data=results.scores[["agent_type", "level", "score"]],
                x="level", y="score", hue="agent_type")
    plt.plot([0.0] * len(results.scores["level"].unique()), "b--")
    plt.show()
    