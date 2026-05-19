from AS0 import AS0

class AS0_log(AS0):
    
    def on_negotiation_success(self, contract, mechanism):
        print(f"success \n{contract}\n")
        
    def first_proposals(self):
        offer = super().first_proposals()
        print("offer\n", offer)
        return offer