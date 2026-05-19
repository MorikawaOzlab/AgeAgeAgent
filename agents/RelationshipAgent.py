from scml.std.agent import StdSyncAgent
from negmas import SAOResponse, ResponseType
from scml.oneshot.common import QUANTITY, TIME, UNIT_PRICE
import random

class RelationshipAgent(StdSyncAgent):
    # def __init__(self):
    n_first_negotiator = 10
    def _step(self) -> int:
        return self.awi.current_step

    def _n_steps(self) -> int:
        return self.awi.n_steps
    def is_supplier(self, partner):
        return partner in self.awi.my_suppliers
    def first_proposals(self):
        partners = list(self.negotiators.keys())
        selected_partners = random.sample(partners, min(self.n_first_negotiator, len(partners)))
        
        responses = {}
        for partner in partners:

            nmi = self.get_nmi(partner)
            if not nmi:
                responses[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                continue

            quantity = nmi.issues[QUANTITY].max_value
            time = nmi.issues[TIME].min_value+1
            unit_price = nmi.issues[UNIT_PRICE].min_value

            offer = (quantity, time, unit_price)
            
            responses[partner] = offer
        return responses
    def counter_all(self, offers, states):
        print(self.awi.needed_sales)
            # return {
            #     partner: SAOResponse(ResponseType.ACCEPT_OFFER, None)
            #     for partner in offers.keys()
            # }
        responses = {}
        for partner in offers.keys():
            nmi = self.get_nmi(partner)
            if not nmi:
                responses[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                continue

            quantity = nmi.issues[QUANTITY].min_value
            time = nmi.issues[TIME].min_value+1
            unit_price = nmi.issues[UNIT_PRICE].min_value

            offer = (quantity, time, unit_price)
            responses[partner] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
        return responses