import hashlib
import json
import os
import re
import time
from pathlib import Path


def get_unique_seed(process_num=None, episode_idx=0):
    """Generate a unique seed using process number, episode index, and high-resolution time."""
    pid = os.getpid()
    time_ns = time.time_ns()
    unique_str = f"{pid}_{process_num}_{episode_idx}_{time_ns}"
    hashed = hashlib.sha256(unique_str.encode()).hexdigest()
    seed = int(hashed[:8], 16)
    return seed


def frame_to_grid(frame):
    """Convert a FrameDataRaw's numpy grid(s) into a JSON-serializable nested list.

    `frame.frame` is a list of numpy arrays (one per game "layer"/sub-frame); we take
    the most recent one, which is what's currently on screen. Each cell is a small
    integer color index (observed range 0-15 across games).
    """
    if not frame.frame:
        return []
    return frame.frame[-1].tolist()


def parse_agent_action(response_str, available_action_names):
    """Parse the task agent's free-form text response into a game action.

    The initial task agent (task_agent.py) only knows how to return a single
    "response" string from one FM call, so this parser has to be permissive: it
    looks for a structured JSON blob first, then falls back to scanning for an
    action-name token, then falls back to a safe default. As the hyperagent evolves
    task_agent.py for this domain, it is free to make the response format stricter.

    Returns:
        (action_name: str, data: dict, valid: bool)
        `data` is only populated (with "x"/"y") for ACTION6.
    """
    available_action_names = available_action_names or ["RESET"]
    text = response_str if isinstance(response_str, str) else str(response_str)

    action_name = None
    data = {}

    # 1) Try a structured JSON blob, e.g. {"action": "ACTION6", "x": 12, "y": 34}
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            candidate = str(parsed.get("action", parsed.get("response", ""))).strip().upper()
            if candidate in available_action_names:
                action_name = candidate
            if "x" in parsed and "y" in parsed:
                data = {"x": int(parsed["x"]), "y": int(parsed["y"])}
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # 2) Fall back to scanning for an available action-name token in the raw text
    if action_name is None:
        upper_text = text.upper()
        for name in available_action_names:
            if re.search(rf"\b{name}\b", upper_text):
                action_name = name
                break

    # 3) If the chosen/only action is ACTION6 and we don't have coordinates yet,
    #    look for the first two integers in [0, 63] mentioned in the text.
    if action_name == "ACTION6" and not data:
        nums = [int(n) for n in re.findall(r"\d+", text)]
        nums = [n for n in nums if 0 <= n <= 63]
        if len(nums) >= 2:
            data = {"x": nums[0], "y": nums[1]}
        else:
            data = {"x": 32, "y": 32}  # center of grid as a harmless default

    if action_name is not None:
        return action_name, data, True

    # 4) Nothing recognizable: fall back to RESET if that's all that's available,
    #    otherwise deterministically pick the first available action.
    fallback = "RESET" if "RESET" in available_action_names and len(available_action_names) == 1 else available_action_names[0]
    if fallback == "ACTION6":
        data = {"x": 32, "y": 32}
    return fallback, data, False


def collect_and_summarize_results(output_dir):
    """Read the raw scorecard + per-episode logs written by harness_arc_agi3 and
    produce report.json with the domain-standard score key ("average_score").

    Unlike the CSV/label-matching domains, there is no ground truth here: the score
    comes directly from the ARC engine's own scorecard (level completion, weighted by
    level order and discounted by action-efficiency vs. a human baseline), computed
    server-side and persisted to raw_scorecard.json at the end of the harness run.
    """
    raw_scorecard_path = os.path.join(output_dir, "raw_scorecard.json")
    with open(raw_scorecard_path, "r") as f:
        raw_scorecard = json.load(f)

    per_game = {}
    for env in raw_scorecard.get("environments", []):
        per_game[env["id"]] = {
            "score": env.get("score", 0.0),
            "levels_completed": env.get("levels_completed", 0),
            "level_count": env.get("level_count", 0),
            "completed": env.get("completed", False),
            "actions": env.get("actions", 0),
        }

    # Collect diagnostics (failed action candidates, step counts) from per-episode logs
    failed_action_candidates = []
    total_steps = 0
    num_episodes = 0
    arc_dir = os.path.join(output_dir, "arc_agi3")
    if os.path.isdir(arc_dir):
        for root, _dirs, files in os.walk(arc_dir):
            for filename in files:
                if filename.endswith(".json"):
                    with open(os.path.join(root, filename), "r") as f:
                        episode_log = json.load(f)
                    failed_action_candidates += episode_log.get("failed_candidates", [])
                    total_steps += episode_log.get("num_steps", 0)
                    num_episodes += 1

    summary = {
        "average_score": raw_scorecard.get("score", 0.0),
        "total_environments": raw_scorecard.get("total_environments", 0),
        "total_environments_completed": raw_scorecard.get("total_environments_completed", 0),
        "total_levels_completed": raw_scorecard.get("total_levels_completed", 0),
        "total_levels": raw_scorecard.get("total_levels", 0),
        "total_actions": raw_scorecard.get("total_actions", 0),
        "average_steps_per_episode": (total_steps / num_episodes) if num_episodes else 0.0,
        "episodes_played": num_episodes,
        "per_game": per_game,
        "failed_action_candidates": failed_action_candidates,
    }

    summary_filename = os.path.join(output_dir, "report.json")
    with open(summary_filename, "w") as f:
        json.dump(summary, f, indent=4)
    return summary


def print_summary_table(summary):
    print("\nSummary of Results:")
    print(f"Overall Average Score: {summary['average_score']:.3f}")
    print(
        f"Levels completed: {summary['total_levels_completed']} / {summary['total_levels']}"
        f"  |  Environments completed: {summary['total_environments_completed']} / {summary['total_environments']}"
        f"  |  Total actions: {summary['total_actions']}"
    )
    print("Per-Game Results:")
    for game_id, game_data in summary["per_game"].items():
        print(
            f"  {game_id}: score={game_data['score']:.3f}, "
            f"levels={game_data['levels_completed']}/{game_data['level_count']}, "
            f"completed={game_data['completed']}, actions={game_data['actions']}"
        )
