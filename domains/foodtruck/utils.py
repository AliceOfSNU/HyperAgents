"""Shared helpers for the foodtruck domain: extracting a single clean command
from the task agent's (possibly chatty) text response, and collecting
per-episode JSON logs into report.json.
"""
import json
import os


def extract_command(response_str):
    """Pull a single clean command line out of a possibly chatty model
    response. Adapted from 11-766-hw2's own simple_llm_agent.py
    SimpleLLMAgent._extract_action -- the env itself validates/parses the
    resulting command (invalid text just costs a prep op, no crash, see
    foodtruck_env.py's own parse_command), so this only needs to strip the
    obvious decorations models tend to add, not fully validate syntax. As
    task_agent.py evolves for this domain it's free to make its own output
    format stricter -- this stays permissive on purpose."""
    if not response_str:
        return "check storage"
    text = response_str if isinstance(response_str, str) else str(response_str)

    cleaned = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("```"):
            continue
        cleaned.append(stripped)
    if not cleaned:
        return "check storage"

    action = cleaned[-1]
    # A single pass only strips one layer -- confirmed live this left
    # "*Command:** check storage" behind for a markdown-bold-labeled
    # response ("**Command:** check storage"), since stripping one leading
    # "*" doesn't expose the "command:" prefix underneath until a second
    # pass. Loop until a full pass finds nothing left to strip.
    while True:
        lowered = action.lower()
        for prefix in (">>>", ">", "-", "*", "action:", "command:", "next:"):
            if lowered.startswith(prefix):
                action = action[len(prefix):].strip()
                break
        else:
            break
    action = action.strip("`").strip().strip('"').strip("'").strip()
    return action.rstrip(".") or "check storage"


def collect_and_summarize_results(output_dir):
    """Read the per-episode JSON logs written by EvaluatorManager and produce
    report.json with the domain-standard score key ("average_roi").

    Score is return-on-investment (cumulative net profit / start_funds),
    not raw dollars -- raw profit isn't comparable across EnvConfig presets
    (different start_funds/horizon_days) and isn't on a similar scale to
    other domains' roughly-[0,1] scores if this domain is ever run alongside
    them for combined parent-selection scoring (see utils/gl_utils.py's
    select_parent, a plain average across every active domain's own score).
    Raw cumulative_reward is still kept per-episode for human readability."""
    per_episode = {}
    rois = []
    total_steps = 0
    total_invalid = 0
    num_episodes = 0

    episodes_dir = os.path.join(output_dir, "foodtruck")
    if os.path.isdir(episodes_dir):
        for filename in sorted(os.listdir(episodes_dir)):
            # EvaluatorManager._run_one also writes seed_<N>_chathistory.md/
            # .json (the raw trajectory, via TaskAgent.save_trajectory -- a
            # JSON *list*, not the episode-log dict this loop expects) into
            # the SAME directory as the seed_<N>.json episode logs -- must
            # exclude those explicitly, not just filter on ".json".
            if not filename.endswith(".json") or filename.endswith("_chathistory.json"):
                continue
            with open(os.path.join(episodes_dir, filename), "r") as f:
                episode_log = json.load(f)
            key = episode_log.get("episode_id", filename)
            per_episode[key] = {
                "roi": episode_log.get("roi", 0.0),
                "cumulative_reward": episode_log.get("cumulative_reward", 0.0),
                "final_funds": episode_log.get("final_funds", 0.0),
                "days_survived": episode_log.get("days_survived", 0),
                "num_steps": episode_log.get("num_steps", 0),
                "invalid_actions": episode_log.get("invalid_actions", 0),
                "bankrupt": episode_log.get("bankrupt", False),
            }
            rois.append(episode_log.get("roi", 0.0))
            total_steps += episode_log.get("num_steps", 0)
            total_invalid += episode_log.get("invalid_actions", 0)
            num_episodes += 1

    summary = {
        "average_roi": (sum(rois) / len(rois)) if rois else 0.0,
        "episodes_played": num_episodes,
        "average_steps_per_episode": (total_steps / num_episodes) if num_episodes else 0.0,
        "average_invalid_actions_per_episode": (total_invalid / num_episodes) if num_episodes else 0.0,
        "per_episode": per_episode,
    }

    summary_filename = os.path.join(output_dir, "report.json")
    with open(summary_filename, "w") as f:
        json.dump(summary, f, indent=4)
    return summary


def print_summary_table(summary):
    print("\nSummary of Results:")
    print(f"Average ROI (net profit / start funds): {summary['average_roi']:.3f}")
    print(
        f"Episodes played: {summary['episodes_played']}"
        f"  |  Avg steps/episode: {summary['average_steps_per_episode']:.1f}"
        f"  |  Avg invalid actions/episode: {summary['average_invalid_actions_per_episode']:.1f}"
    )
    print("Per-Episode Results:")
    for episode_id, data in summary["per_episode"].items():
        print(
            f"  {episode_id}: roi={data['roi']:.3f}, reward={data['cumulative_reward']:.2f}, "
            f"final_funds={data['final_funds']:.2f}, days={data['days_survived']}, "
            f"steps={data['num_steps']}, invalid={data['invalid_actions']}, bankrupt={data['bankrupt']}"
        )
