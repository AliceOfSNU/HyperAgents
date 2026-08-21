"""Pier agent for skills/branching: a fresh deep-swe task container that
replays a recorded trajectory's prefix, then continues with fresh sampling.
See branch_task_agent.py (what actually runs inside the container) and
skills/branching/SKILL.md (the full picture).

Subclasses HyperAgentsSweAgent (pier_agent.py) rather than duplicating it --
setup() (baseline code + wheels upload) and network_allowlist() are genuinely
shared, code-version-independent concerns; only run() differs (a different
container-side entrypoint, extra --ak kwargs, no chat_history_file name
collision with a sibling branch running the same task in parallel)."""

import shlex
import uuid
from pathlib import Path

from .config import AGENT_MODEL
from .pier_agent import CODE_DIR, REPO_DIR, HyperAgentsSweAgent


class BranchRolloutPierAgent(HyperAgentsSweAgent):
    def __init__(
        self,
        logs_dir,
        root_dir=None,
        model_name=None,
        agent_timeout_sec=None,
        source_chat_history_path=None,
        branch_round=None,
        original_max_tool_calls=100,
        continuation_max_tool_calls=20,
        temperature=0.7,
        **kwargs,
    ):
        super().__init__(
            logs_dir=logs_dir, root_dir=root_dir, model_name=model_name or AGENT_MODEL,
            agent_timeout_sec=agent_timeout_sec, **kwargs,
        )
        if not source_chat_history_path or branch_round is None:
            raise ValueError(
                "BranchRolloutPierAgent requires --ak source_chat_history_path=<...> "
                "and --ak branch_round=<N> -- see skills/branching/SKILL.md."
            )
        self._source_chat_history_path = source_chat_history_path
        self._branch_round = int(branch_round)
        self._original_max_tool_calls = int(original_max_tool_calls)
        self._continuation_max_tool_calls = int(continuation_max_tool_calls)
        self._temperature = float(temperature)

    @staticmethod
    def name() -> str:
        return "hyperagents-branch-rollout"

    async def setup(self, environment) -> None:
        await super().setup(environment)
        # run_branch_rollout.py / branch_task_agent.py / trajectory_replay.py
        # are the branching skill's OWN harness code -- not part of the
        # per-generation evolvable snapshot BASELINE_FILES copies from
        # self._root_dir, so they're never in there regardless of whether
        # that snapshot happens to predate these files existing at all
        # (confirmed live: this exact gap is what "python: can't open file
        # run_branch_rollout.py" meant the first time this ran). Uploaded
        # explicitly from this file's own fixed repo-root location instead.
        harness_root = Path(__file__).resolve().parents[2]
        await environment.exec(command=f"mkdir -p {CODE_DIR}/domains/deep_swe", user="root")
        await environment.upload_file(
            source_path=harness_root / "run_branch_rollout.py",
            target_path=f"{CODE_DIR}/run_branch_rollout.py",
        )
        await environment.upload_file(
            source_path=harness_root / "domains" / "deep_swe" / "branch_task_agent.py",
            target_path=f"{CODE_DIR}/domains/deep_swe/branch_task_agent.py",
        )
        await environment.upload_file(
            source_path=harness_root / "agent" / "trajectory_replay.py",
            target_path=f"{CODE_DIR}/agent/trajectory_replay.py",
        )

    async def run(self, instruction, environment, context) -> None:
        instruction_path = f"{CODE_DIR}/instruction_{uuid.uuid4().hex[:8]}.md"
        heredoc_marker = f"HA_INSTR_EOF_{uuid.uuid4().hex[:8]}"
        write_cmd = (
            f"cat > {shlex.quote(instruction_path)} << '{heredoc_marker}'\n"
            f"{instruction}\n"
            f"{heredoc_marker}\n"
        )
        await environment.exec(command=write_cmd)

        source_chat_history_target = f"{CODE_DIR}/source_chat_history.json"
        await environment.upload_file(
            source_path=self._source_chat_history_path, target_path=source_chat_history_target,
        )

        container_chat_history = f"{CODE_DIR}/chat_history.md"
        env = {}
        for key in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
            value = self._get_env(key)
            if value:
                env[key] = value

        timeout_sec = int(self._agent_timeout_sec) if self._agent_timeout_sec else None
        result = await environment.exec(
            command=(
                f"cd {CODE_DIR} && python run_branch_rollout.py "
                f"--repo_dir {shlex.quote(REPO_DIR)} "
                f"--instruction_file {shlex.quote(instruction_path)} "
                f"--chat_history_file {shlex.quote(container_chat_history)} "
                f"--model {shlex.quote(self.model_name)} "
                f"--source_chat_history {shlex.quote(source_chat_history_target)} "
                f"--branch_round {self._branch_round} "
                f"--original_max_tool_calls {self._original_max_tool_calls} "
                f"--continuation_max_tool_calls {self._continuation_max_tool_calls} "
                f"--temperature {self._temperature}"
            ),
            env=environment.agent_process_env(env),
            timeout_sec=timeout_sec,
        )
        if result.return_code != 0:
            self.logger.warning(
                f"branch rollout exited {result.return_code}\n"
                f"stdout: {(result.stdout or '')[:3000]}\n"
                f"stderr: {(result.stderr or '')[:3000]}"
            )

        for suffix in (".md", ".json"):
            src = container_chat_history.rsplit(".", 1)[0] + suffix
            try:
                await environment.download_file(
                    source_path=src, target_path=self.logs_dir / Path(src).name,
                )
            except Exception as exc:
                self.logger.debug(f"Could not download {src}: {exc}")
