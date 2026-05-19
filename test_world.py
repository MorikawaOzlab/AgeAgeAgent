import pandas as pd
import math
from typing import Iterable
from rich.jupyter import print

from negmas import SAOResponse, ResponseType, Outcome, SAOState
from scml.std import *
from scml.runner import WorldRunner
import matplotlib.pyplot as plt

# create a runner that encapsulates a number of configs to evaluate agents
# in the same conditions every time

class MyAgent(StdAgent):
    def __init__(self, *args, **kwargs):
        self.internal_dict = {}
        self.count = 0
        super().__init__(*args, **kwargs)
    def before_step(self):
        if self.count % 2 == 1: 
            self.internal_dict["balance"] = int(self.awi.current_balance)
            self.internal_dict["inventory"] = int(self.awi.current_step)
            self.internal_dict["inveaaantory"] = int(self.awi.current_score)
            # self.internal_dict["inventory"] = tuple(map(int, self.awi.current_inventory))

            self.awi.logdebug_agent(f"internal_dict: {self.internal_dict}")
        self.count += 1
    def propose(self, negotiator_id, state):
        return super().propose(negotiator_id, state)
    
    def respond(self, negotiator_id, state, source=None):
        return ResponseType.ACCEPT_OFFER


CONFIGS, REPS, STEPS = 1, 3, 10
context = ANACStdContext(
    n_steps=STEPS, n_processes=3, world_params=dict(construct_graphs=True)
)
single_agent_runner = WorldRunner(
    context, n_configs=CONFIGS, n_repetitions=REPS, save_worlds=True
)
full_market_runner = WorldRunner.from_runner(
    single_agent_runner, control_all_agents=True
)

agent_type = [MyAgent, RandomStdAgent]
full_market_runner(MyAgent);
full_market_runner(RandomStdAgent);

full_market_runner.draw_worlds_of(MyAgent);
full_market_runner(RandomStdAgent);

full_market_runner.plot_stats(agg=False);
plt.show()

full_market_runner.score_summary()