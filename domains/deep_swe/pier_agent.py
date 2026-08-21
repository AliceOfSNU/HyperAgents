"""Custom Pier agent that runs HyperAgents' own swe_task_agent.py inside a
deep-swe task container, instead of one of Pier's built-in agents
(mini-swe-agent, claude-code, etc.).

Mirrors mini_swe_agent.py's own shape (github.com/datacurve-ai/pier ->
pier/agents/installed/mini_swe_agent.py): setup() uploads code and installs
deps, run() makes exactly ONE environment.exec() call that lets the whole
interactive agent loop run standalone inside the container (writing its own
chat_history.md/.json via AgentSystem.save_trajectory), then downloads those
files back so they land alongside Pier's own trial artifacts. The agent loop
itself (LLM calls, tool execution, Plan-Act-Observe rounds) is 100% our own
existing code, unchanged -- Pier only owns container lifecycle, patch
capture, and grading.

`root_dir` (an --agent-kwarg / --ak flag) must point at a host directory
already reconstructed to the exact generation being evaluated -- see
domains/deep_swe/harness.py, which applies that generation's patch chain
host-side before ever invoking `pier run`. This module never does patch
reconstruction itself.
"""

import os
import shlex
import uuid
from pathlib import Path

from pier.agents.base import BaseAgent
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.network import NetworkAllowlist

from .config import AGENT_MODEL, BASELINE_FILES, PROVIDER_API_DOMAINS

REPO_DIR = "/app"  # Harbor's own convention -- confirmed live via task.toml's
                    # own [[verifier.collect]] hook, which does `cd /app`.
CODE_DIR = "/hyperagents_code"


class HyperAgentsSweAgent(BaseAgent):
    SUPPORTS_WINDOWS: bool = False

    def __init__(
        self,
        logs_dir: Path,
        root_dir: str | None = None,
        model_name: str | None = None,
        agent_timeout_sec: float | None = None,
        **kwargs,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name or AGENT_MODEL, **kwargs)
        if not root_dir:
            raise ValueError(
                "HyperAgentsSweAgent requires --ak root_dir=<reconstructed gen_N code dir> "
                "-- see domains/deep_swe/harness.py."
            )
        self._root_dir = Path(root_dir)
        self._agent_timeout_sec = agent_timeout_sec

    @staticmethod
    def name() -> str:
        return "hyperagents-swe"

    def version(self) -> str:
        return "0.1.0"

    def network_allowlist(self) -> NetworkAllowlist:
        # Deliberately narrow -- see this package's config.py docstring for
        # why (deep-swe's own solution.patch files are public on GitHub;
        # this allowlist, not local file hiding, is the real defense).
        provider = self.model_name.split("/", 1)[0] if self.model_name and "/" in self.model_name else None
        return NetworkAllowlist(domains=PROVIDER_API_DOMAINS.get(provider or "", []))

    async def setup(self, environment: BaseEnvironment) -> None:
        await environment.exec(command=f"mkdir -p {CODE_DIR}", user="root")
        for rel in BASELINE_FILES:
            src = self._root_dir / rel.rstrip("/")
            if not src.exists():
                continue
            target = f"{CODE_DIR}/{rel.rstrip('/')}"
            if src.is_dir():
                await environment.exec(command=f"mkdir -p {target}", user="root")
                await environment.upload_dir(source_dir=src, target_dir=target)
            else:
                await environment.upload_file(source_path=src, target_path=target)

        # Installed from wheels pre-fetched host-side by domains/deep_swe/
        # harness.py's _fetch_wheels(), with --no-index -- the container
        # never touches PyPI (or any network) for its own dependencies, only
        # the LLM provider's own API domain during run(). See this module's
        # own docstring and harness.py's _fetch_wheels() for why.
        wheels_src = self._root_dir / "_wheels"
        req_path = f"{CODE_DIR}/requirements-deep_swe.txt"
        if wheels_src.is_dir() and any(wheels_src.iterdir()):
            wheels_target = f"{CODE_DIR}/_wheels"
            await environment.exec(command=f"mkdir -p {wheels_target}", user="root")
            await environment.upload_dir(source_dir=wheels_src, target_dir=wheels_target)
            result = await environment.exec(
                command=f"pip install -q --no-index --find-links={wheels_target} -r {req_path}",
                timeout_sec=300,
            )
            if result.return_code != 0:
                self.logger.warning(
                    f"Offline pip install failed: {(result.stderr or '')[:2000]}"
                )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        task_id = environment.environment_name
        instruction_path = f"{CODE_DIR}/instruction_{uuid.uuid4().hex[:8]}.md"
        heredoc_marker = f"HA_INSTR_EOF_{uuid.uuid4().hex[:8]}"
        write_cmd = (
            f"cat > {shlex.quote(instruction_path)} << '{heredoc_marker}'\n"
            f"{instruction}\n"
            f"{heredoc_marker}\n"
        )
        await environment.exec(command=write_cmd)

        container_chat_history = f"{CODE_DIR}/chat_history.md"
        env = {}
        for key in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
            value = self._get_env(key)
            if value:
                env[key] = value

        timeout_sec = int(self._agent_timeout_sec) if self._agent_timeout_sec else None
        result = await environment.exec(
            command=(
                f"cd {CODE_DIR} && python run_swe_task_agent.py "
                f"--repo_dir {shlex.quote(REPO_DIR)} "
                f"--instruction_file {shlex.quote(instruction_path)} "
                f"--task_id {shlex.quote(task_id)} "
                f"--chat_history_file {shlex.quote(container_chat_history)} "
                f"--model {shlex.quote(self.model_name)}"
            ),
            # agent_process_env merges in the egress-proxy's HTTPS_PROXY +
            # auth token env vars (see network_allowlist()) -- without this,
            # litellm inside the container tries a direct connection to the
            # LLM API and fails, since the container has no direct internet
            # access at all, only through the proxy (confirmed live: a bare
            # env=env here produced "Temporary failure in name resolution").
            env=environment.agent_process_env(env),
            timeout_sec=timeout_sec,
        )
        if result.return_code != 0:
            self.logger.warning(
                f"swe_task_agent exited {result.return_code} for {task_id}\n"
                f"stdout: {(result.stdout or '')[:3000]}\n"
                f"stderr: {(result.stderr or '')[:3000]}"
            )

        for suffix in (".md", ".json"):
            src = container_chat_history.rsplit(".", 1)[0] + suffix
            try:
                await environment.download_file(
                    source_path=src,
                    target_path=self.logs_dir / Path(src).name,
                )
            except Exception as exc:
                self.logger.debug(f"Could not download {src}: {exc}")

    def _get_env(self, key: str) -> str | None:
        return os.environ.get(key)
