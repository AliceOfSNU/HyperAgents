import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig

from domains.arc_agi3.agents import AgentFactory
from domains.arc_agi3.evaluator import EvaluatorManager
from domains.arc_agi3.utils import collect_and_summarize_results, print_summary_table


@contextmanager
def redirect_to_file(filepath):
    original = sys.stdout
    with open(filepath, "w") as file:
        sys.stdout = file
        try:
            yield
        finally:
            sys.stdout = original


def harness_arc_agi3(config):
    # Determine output directory
    if config.eval.resume_from is not None:
        output_dir = config.eval.resume_from
    else:
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f") if config.eval.run_id is None else config.eval.run_id
        output_dir = os.path.join(config.eval.output_dir, timestamp)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Setup logger
    log_filename = os.path.join(output_dir, "eval.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename)],
        force=True,
    )

    evaluator_manager = EvaluatorManager(config, output_dir=output_dir)
    agent_factory = AgentFactory(config)
    with redirect_to_file(log_filename):
        evaluator_manager.run(agent_factory)

    return output_dir


def report_arc_agi3(output_dir):
    summary = collect_and_summarize_results(output_dir)
    print_summary_table(summary)
    return summary


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    output_dir = harness_arc_agi3(config)
    report_arc_agi3(output_dir)


if __name__ == "__main__":
    main()
