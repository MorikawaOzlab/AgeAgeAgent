from scml.std.agent import StdSyncAgent
from negmas import SAOResponse, ResponseType
from scml.oneshot.common import QUANTITY, TIME, UNIT_PRICE
import random

class RelationshipAgent(StdSyncAgent):
    n_first_negotiator = 3
    # def __init__(self):
    def _step(self) -> int:
        return self.awi.current_step

    def _n_steps(self) -> int:
        return self.awi.n_steps

    def first_proposals(self):
        partners = list(self.awi.current_states.keys())
        selected_partners = random.sample(partners, min(self.n_first_negotiator, len(partners)))
        
        responses = {}
        for partner in selected_partners:
            nmi = self.get_nmi(partner)
            if not nmi:
                responses[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                continue

            quantity = (nmi.issues[QUANTITY].max_value + nmi.issues[QUANTITY].min_value) // 2
            time = nmi.issues[TIME].min_value+1
            unit_price = nmi.issues[UNIT_PRICE].min_value

            offer = (quantity, time, unit_price)
            
            responses[partner] = SAOResponse(ResponseType.REJECT_OFFER, offer)
            print(offer)
        return responses
    def counter_all(self, offers, states):
        print("counter_all")
        return {
            partner: SAOResponse(ResponseType.ACCEPT_OFFER, None)
            for partner in offers.keys()
        }    
        # responses = {}
        # for partner in offers.keys():
        #     if states[partner].current_proposer == self.id: 
        #         print("ok")
        #         responses[partner] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
        #     else:
        #         print("not ok")
        #         responses[partner] = SAOResponse(ResponseType.END_NEGOTIATION, None)
        # return responses