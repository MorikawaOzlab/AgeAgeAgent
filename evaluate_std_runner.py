from __future__ import annotations

from scml.std import ANACStdContext, GreedyStdAgent
from scml.runner import WorldRunner

# 自分のエージェントをここでimport
from AgeAgeAgent import AgeAgeAgent
# AgeAgeAgent = GreedyStdAgent


def main() -> None:
    """ANAC Standardに近い条件で、自分のエージェントを複数worldで評価する。"""

    configs = 10
    reps = 3
    steps = 50

    context = ANACStdContext(
        n_steps=steps,
        n_processes=3,
        world_params=dict(construct_graphs=True),
    )

    runner = WorldRunner(
        context,
        n_configs=configs,
        n_repetitions=reps,
        save_worlds=True,
    )

    runner(AgeAgeAgent)

    print("=== score summary ===")
    print(runner.score_summary())

    # 必要ならグラフ表示
    # runner.plot_stats(stats="score", agg=False)


if __name__ == "__main__":
    main()