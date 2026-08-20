import json
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from domains.arc_agi3.utils import frame_to_grid, get_unique_seed, parse_agent_action

logger = logging.getLogger(__name__)


class EvaluatorManager:
    """Manages evaluation of a task agent across one or more ARC-AGI-3 games.

    Mirrors domains/balrog/evaluator.py's EvaluatorManager, but there is only one
    "environment type" here (the ARC engine itself); the games to play come from
    config.tasks.arc_agi3_tasks, each played for config.eval.num_episodes.arc_agi3
    episodes.
    """

    def __init__(self, config, output_dir="."):
        import arc_agi

        self.config = config
        self.output_dir = output_dir
        self.game_ids = list(config.tasks.arc_agi3_tasks)
        self.num_episodes = config.eval.num_episodes.arc_agi3
        self.num_workers = config.eval.num_workers
        self.max_steps = config.eval.max_steps_per_episode
        self.max_resets = config.eval.max_resets_per_episode

        arc_api_key = config.envs.get("arc_api_key") or os.environ.get("ARC_API_KEY", "")
        operation_mode_str = config.envs.get("operation_mode", "normal")
        self.arc = arc_agi.Arcade(
            arc_api_key=arc_api_key,
            arc_base_url=config.envs.get("arc_base_url", "https://three.arcprize.org"),
            operation_mode=arc_agi.OperationMode(operation_mode_str),
        )
        self.scorecard_id = self.arc.create_scorecard(
            tags=["hyperagents", "dgm-h"],
        )

        self.tasks = []
        for game_id in self.game_ids:
            for episode_idx in range(self.num_episodes):
                json_filename = os.path.join(
                    self.output_dir, "arc_agi3", game_id, f"{game_id}_run_{episode_idx:02d}.json"
                )
                if os.path.exists(json_filename):
                    logging.info(f"Skipping completed episode: {game_id}, episode {episode_idx}")
                else:
                    self.tasks.append((game_id, episode_idx))

    def run(self, agent_factory):
        results = []
        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(self._run_one, agent_factory, game_id, episode_idx): (game_id, episode_idx)
                    for game_id, episode_idx in self.tasks
                }
                for future in as_completed(futures):
                    game_id, episode_idx = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as e:
                        logging.error(f"Error evaluating {game_id} episode {episode_idx}: {e}")
        else:
            for game_id, episode_idx in self.tasks:
                try:
                    results.append(self._run_one(agent_factory, game_id, episode_idx))
                except Exception as e:
                    logging.error(f"Error evaluating {game_id} episode {episode_idx}: {e}")

        # Finalize the scorecard and persist it so report_arc_agi3 (a separate
        # process, run later) doesn't need to re-contact the ARC API at all.
        scorecard = self.arc.close_scorecard(self.scorecard_id)
        if scorecard is None:
            scorecard = self.arc.get_scorecard(self.scorecard_id)
        raw_scorecard_path = os.path.join(self.output_dir, "raw_scorecard.json")
        with open(raw_scorecard_path, "w") as f:
            f.write(scorecard.model_dump_json(indent=2) if scorecard is not None else "{}")

        return results

    def _run_one(self, agent_factory, game_id, episode_idx):
        seed = self.config.envs.env_kwargs.seed
        if seed is None:
            seed = get_unique_seed(episode_idx=episode_idx)
        random.seed(seed)
        np.random.seed(seed)

        chat_history_file = os.path.join(
            self.output_dir, "arc_agi3", game_id, f"{game_id}_run_{episode_idx:02d}_chathistory.md"
        )
        Path(chat_history_file).parent.mkdir(exist_ok=True, parents=True)
        agent = agent_factory.create_agent(chat_history_file=chat_history_file)

        episode_log = run_episode(
            self.arc,
            game_id=game_id,
            agent=agent,
            seed=seed,
            max_steps=self.max_steps,
            max_resets=self.max_resets,
            scorecard_id=self.scorecard_id,
            feedback_on_invalid_action=self.config.eval.feedback_on_invalid_action,
        )

        json_filename = os.path.join(
            self.output_dir, "arc_agi3", game_id, f"{game_id}_run_{episode_idx:02d}.json"
        )
        Path(json_filename).parent.mkdir(exist_ok=True, parents=True)
        with open(json_filename, "w") as f:
            json.dump(episode_log, f, indent=4)

        return episode_log


def run_episode(arc, game_id, agent, seed, max_steps, max_resets, scorecard_id=None, feedback_on_invalid_action=True):
    """Play one episode of `game_id` with `agent`, one discrete GameAction per step.

    At each step the task agent is given the current grid, its available actions,
    and whether its previous action was understood, and must respond with a chosen
    action (see domains/arc_agi3/utils.py:parse_agent_action for the response
    format it's expected to produce). The episode ends on WIN, on exhausting the
    step budget, or after too many GAME_OVER->RESET cycles.

    NOTE: scorecard_id must be passed explicitly. If omitted, arc.make() silently
    creates and uses its own separate default scorecard, so runs would go
    unrecorded on the scorecard this evaluator later reads the score back from.
    """
    from arcengine import GameAction, GameState

    env = arc.make(game_id, seed=seed, scorecard_id=scorecard_id)
    frame = env.reset()

    episode_log = {"game_id": game_id, "seed": seed, "steps": [], "failed_candidates": []}
    prev_action_name, prev_valid = None, True
    resets = 0
    step = 0

    for step in range(max_steps):
        available_action_names = [GameAction.from_id(a).name for a in frame.available_actions] or ["RESET"]

        inputs = {
            "domain": "arc_agi3",
            "game_id": game_id,
            "state": frame.state.name,
            "levels_completed": frame.levels_completed,
            "win_levels": frame.win_levels,
            "grid": frame_to_grid(frame),
            "grid_legend": "64x64 grid; each cell is an integer color index (0-15)",
            "available_actions": available_action_names,
            "step": step,
            "prev_action": prev_action_name,
            "prev_action_understood": prev_valid if feedback_on_invalid_action else None,
        }

        response_str, _ = agent.forward(inputs)
        action_name, data, valid = parse_agent_action(response_str, available_action_names)
        if not valid:
            episode_log["failed_candidates"].append(str(response_str)[:500])

        action = GameAction.from_name(action_name)
        step_kwargs = {"data": data} if action.is_complex() else {}
        frame = env.step(action, reasoning={"step": step, "response": str(response_str)[:500]}, **step_kwargs)

        episode_log["steps"].append(
            {
                "step": step,
                "action": action_name,
                "data": data,
                "valid": valid,
                "state": frame.state.name,
                "levels_completed": frame.levels_completed,
            }
        )
        prev_action_name, prev_valid = action_name, valid

        if frame.state == GameState.WIN:
            break
        if frame.state == GameState.GAME_OVER:
            resets += 1
            if resets > max_resets:
                break
            frame = env.reset()

    episode_log["num_steps"] = step + 1
    episode_log["final_state"] = frame.state.name
    episode_log["levels_completed"] = frame.levels_completed
    episode_log["win_levels"] = frame.win_levels
    episode_log["won"] = frame.state == GameState.WIN
    return episode_log
