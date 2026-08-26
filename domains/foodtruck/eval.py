"""FoodTruckEnv (CMU 11-766 HW2's Gymnasium-compatible, text-only food-truck
simulation, vendored under domains/foodtruck/environments/ -- see that
package's own provenance note) wired into this project's harness/report
dispatch, mirroring domains/arc_agi3/eval.py's shape.

Unlike deep_swe/research (Docker/Pier-sandboxed, since their task agents run
bash/editor tools against a real repo), this domain needs no sandboxing --
the environment is a pure, hermetic Python simulation with a text action/
observation interface, so it runs in-process inside the same shared
meta-agent container as the other gym-style domains (arc_agi3/balrog/
genesis), via the generic eval_produced_agent() path in generate_loop.py.
Deliberately cheap and solvable by a small model: short horizon by default
(see config.yaml), a handful of seeds, no external API calls.
"""
import logging
import os
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig

from domains.foodtruck.agents import AgentFactory
from domains.foodtruck.evaluator import EvaluatorManager
from domains.foodtruck.utils import collect_and_summarize_results, print_summary_table


def harness_foodtruck(config):
    if config.eval.resume_from is not None:
        output_dir = config.eval.resume_from
    else:
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f") if config.eval.run_id is None else config.eval.run_id
        output_dir = os.path.join(config.eval.output_dir, timestamp)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    log_filename = os.path.join(output_dir, "eval.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename)],
        force=True,
    )

    evaluator_manager = EvaluatorManager(config, output_dir=output_dir)
    agent_factory = AgentFactory(config)
    evaluator_manager.run(agent_factory)

    return output_dir


def report_foodtruck(output_dir):
    summary = collect_and_summarize_results(output_dir)
    print_summary_table(summary)
    return summary


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    output_dir = harness_foodtruck(config)
    report_foodtruck(output_dir)


if __name__ == "__main__":
    main()
