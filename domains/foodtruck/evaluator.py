import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from domains.foodtruck.environments import EnvConfig, FoodTruckEnv
from domains.foodtruck.utils import extract_command

logger = logging.getLogger(__name__)

RULES = """\
You are the owner of a food truck. Each day has a PREP stage: you buy ingredients, build a
menu of dishes at chosen prices, then `end prep` to open for the day (or prep auto-ends once
you run out of prep operations). Your goal is to maximize cumulative net profit (revenue
minus rent) across the whole run.

Commands (case-insensitive, one prep operation each): check storage | check market |
check recipes | check menu | buy <ingredient> <qty> | trash <ingredient> <qty> |
set menu <slot> <recipe> <price> | clear menu <slot> | end prep
<ingredient>/<recipe> may be a name or 1-based index. `check ...` output only appears in the
reply right after you check it, then disappears -- remember it. Reply with exactly ONE
command on its own line."""


class EvaluatorManager:
    """Manages evaluation of a task agent across one or more FoodTruckEnv
    episodes. Mirrors domains/arc_agi3/evaluator.py's EvaluatorManager -- but
    unlike ARC-AGI-3 (many distinct external games), there is one environment
    here; the "tasks" are distinct random seeds (config.tasks.foodtruck_seeds),
    each giving a different market/demand randomization, run once each."""

    def __init__(self, config, output_dir="."):
        self.config = config
        self.output_dir = output_dir
        self.seeds = list(config.tasks.foodtruck_seeds)
        self.num_workers = config.eval.num_workers
        self.max_steps = config.eval.max_steps_per_episode
        self.env_config = EnvConfig(**dict(config.envs.env_kwargs))

        self.tasks = []
        for seed in self.seeds:
            json_filename = Path(self.output_dir) / "foodtruck" / f"seed_{seed}.json"
            if json_filename.exists():
                logging.info(f"Skipping completed episode: seed {seed}")
            else:
                self.tasks.append(seed)

    def run(self, agent_factory):
        results = []
        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(self._run_one, agent_factory, seed): seed for seed in self.tasks}
                for future in as_completed(futures):
                    seed = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as e:
                        logging.error(f"Error evaluating seed {seed}: {e}")
        else:
            for seed in self.tasks:
                try:
                    results.append(self._run_one(agent_factory, seed))
                except Exception as e:
                    logging.error(f"Error evaluating seed {seed}: {e}")
        return results

    def _run_one(self, agent_factory, seed):
        chat_history_file = Path(self.output_dir) / "foodtruck" / f"seed_{seed}_chathistory.md"
        chat_history_file.parent.mkdir(exist_ok=True, parents=True)
        agent = agent_factory.create_agent(chat_history_file=str(chat_history_file))

        env = FoodTruckEnv(self.env_config)
        episode_log = run_episode(env, agent=agent, seed=seed, max_steps=self.max_steps)

        json_filename = Path(self.output_dir) / "foodtruck" / f"seed_{seed}.json"
        json_filename.parent.mkdir(exist_ok=True, parents=True)
        with open(json_filename, "w") as f:
            json.dump(episode_log, f, indent=4)

        return episode_log


def run_episode(env, agent, seed, max_steps):
    """Play one FoodTruckEnv episode with `agent`, one text command per step.

    At each step the task agent is given the env's current observation (which
    already includes day/funds/stage and whatever it last `check`ed) plus a
    compact rules cheatsheet, and must respond with a chosen command -- see
    domains/foodtruck/utils.py:extract_command for the response format it's
    expected to produce. The episode ends when the env itself terminates
    (horizon reached, or bankruptcy if EnvConfig.bankrupt_terminates) or the
    step-budget safety cap is hit, whichever comes first."""
    obs, _info = env.reset(seed=seed)

    episode_log = {"episode_id": f"seed_{seed}", "seed": seed, "steps": [], "failed_candidates": []}
    prev_action, prev_reward = None, None
    cumulative_reward = 0.0
    invalid_actions = 0
    step = 0
    terminated = False

    for step in range(max_steps):
        inputs = {
            "domain": "foodtruck",
            "observation": obs,
            "rules": RULES,
            "step": step,
            "prev_action": prev_action,
            "prev_reward": prev_reward,
        }

        response_str, _ = agent.forward(inputs)
        action = extract_command(response_str)

        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        if info.get("parsed_op") == "invalid":
            invalid_actions += 1
            episode_log["failed_candidates"].append(str(response_str)[:500])

        episode_log["steps"].append({
            "step": step, "action": action, "reward": reward,
            "parsed_op": info.get("parsed_op"), "day": info.get("day"),
        })
        prev_action, prev_reward = action, reward

        if terminated or truncated:
            break

    episode_log["num_steps"] = step + 1
    episode_log["days_survived"] = env.day_idx
    episode_log["final_funds"] = env.funds
    episode_log["cumulative_reward"] = cumulative_reward
    episode_log["roi"] = (cumulative_reward / env.config.start_funds) if env.config.start_funds else 0.0
    episode_log["bankrupt"] = env.funds < 0
    episode_log["invalid_actions"] = invalid_actions
    episode_log["hit_step_cap_without_terminating"] = not terminated
    return episode_log
