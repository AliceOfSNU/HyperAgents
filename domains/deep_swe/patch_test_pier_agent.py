"""Pier agent for skills/patch_testing: a fresh deep-swe task container that
replays a recorded trajectory's prefix, then continues using THIS
container's own swe_task_agent.py (see patch_test_task_agent.py). See
skills/patch_testing/SKILL.md for the full picture.

Subclasses HyperAgentsSweAgent (pier_agent.py) rather than duplicating it --
setup()'s BASELINE_FILES upload and network_allowlist() are genuinely
shared; only run() differs (a different container-side entrypoint, extra
--ak kwargs). Crucially, setup() uploads swe_task_agent.py FROM
self._root_dir -- whether that's the checkpoint's original code (control)
or that same code with the meta-agent's patch applied on top (treatment) is
decided entirely by which root_dir the orchestrator passes in; this class
has no idea which arm it's running and doesn't need to."""

import shlex
import uuid
from pathlib import Path

from .config import AGENT_MODEL
from .pier_agent import CODE_DIR, REPO_DIR, HyperAgentsSweAgent


class PatchTestPierAgent(HyperAgentsSweAgent):
    def __init__(
        self,
        logs_dir,
        root_dir=None,
        model_name=None,
        agent_timeout_sec=None,
        source_chat_history_path=None,
        checkpoint_round=None,
        temperature=0.7,
        **kwargs,
    ):
        super().__init__(
            logs_dir=logs_dir, root_dir=root_dir, model_name=model_name or AGENT_MODEL,
            agent_timeout_sec=agent_timeout_sec, **kwargs,
        )
        if not source_chat_history_path or checkpoint_round is None:
            raise ValueError(
                "PatchTestPierAgent requires --ak source_chat_history_path=<...> "
                "and --ak checkpoint_round=<N> -- see skills/patch_testing/SKILL.md."
            )
        self._source_chat_history_path = source_chat_history_path
        self._checkpoint_round = int(checkpoint_round)
        self._temperature = float(temperature)

    @staticmethod
    def name() -> str:
        return "hyperagents-patch-test-rollout"

    async def setup(self, environment) -> None:
        await super().setup(environment)
        # run_patch_test_rollout.py / patch_test_task_agent.py are this
        # skill's OWN harness code -- not part of the per-generation
        # evolvable BASELINE_FILES snapshot (which only ever carries
        # swe_task_agent.py itself, uploaded above by super().setup() from
        # whichever root_dir this specific arm/replicate was given).
        # Uploaded explicitly from this file's own fixed repo-root location,
        # same pattern as skills/branching's branch_pier_agent.py.
        harness_root = Path(__file__).resolve().parents[2]
        await environment.exec(command=f"mkdir -p {CODE_DIR}/domains/deep_swe", user="root")
        await environment.upload_file(
            source_path=harness_root / "run_patch_test_rollout.py",
            target_path=f"{CODE_DIR}/run_patch_test_rollout.py",
        )
        await environment.upload_file(
            source_path=harness_root / "domains" / "deep_swe" / "patch_test_task_agent.py",
            target_path=f"{CODE_DIR}/domains/deep_swe/patch_test_task_agent.py",
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

        task_id = environment.environment_name
        container_chat_history = f"{CODE_DIR}/chat_history.md"
        env = {}
        for key in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
            value = self._get_env(key)
            if value:
                env[key] = value

        timeout_sec = int(self._agent_timeout_sec) if self._agent_timeout_sec else None
        result = await environment.exec(
            command=(
                f"cd {CODE_DIR} && python run_patch_test_rollout.py "
                f"--repo_dir {shlex.quote(REPO_DIR)} "
                f"--instruction_file {shlex.quote(instruction_path)} "
                f"--task_id {shlex.quote(task_id)} "
                f"--chat_history_file {shlex.quote(container_chat_history)} "
                f"--model {shlex.quote(self.model_name)} "
                f"--source_chat_history {shlex.quote(source_chat_history_target)} "
                f"--checkpoint_round {self._checkpoint_round} "
                f"--temperature {self._temperature}"
            ),
            env=environment.agent_process_env(env),
            timeout_sec=timeout_sec,
        )
        if result.return_code != 0:
            self.logger.warning(
                f"patch-test rollout exited {result.return_code}\n"
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
