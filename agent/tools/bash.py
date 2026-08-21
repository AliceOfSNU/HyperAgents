import asyncio
import atexit
import os
import signal
import threading
import uuid


def tool_info():
    return {
        "name": "bash",
        "description": """Run commands in a bash shell
* When invoking this tool, the contents of the "command" parameter does NOT need to be XML-escaped.
* You don't have access to the internet via this tool.
* You do have access to a mirror of common linux and python packages via apt and pip.
* State is persistent across command calls and discussions with the user.
* To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.
* Please avoid commands that may produce a very large amount of output.
* Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run."
                }
            },
            "required": ["command"]
        }
    }


class BashSession:
    """A persistent bash shell.

    The shell is deliberately non-interactive: an interactive shell prints
    prompts/echoes commands to stderr and emits the tcsetattr/ioctl noise
    that the old filter_error had to strip, and none of that is useful to
    the agent. A non-interactive shell still keeps cwd/env/background-job
    state across `run` calls, which is the actual contract the tool
    description promises.
    """

    def __init__(self):
        self._started = False
        self._process = None
        self._timed_out = False
        self._timeout = 120.0  # seconds
        self._output_delay = 0.05  # seconds

    async def start(self):
        if self._started:
            return
        self._process = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "--noprofile",
            "--norc",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),  # Ensures inheritance of the current environment
            preexec_fn=os.setsid,  # Own process group, so stop() can kill children too
        )
        self._started = True

    def stop(self):
        if not self._started:
            return
        proc = self._process
        self._process = None
        self._started = False
        if proc is None:
            return
        if proc.returncode is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        # Close the subprocess transport explicitly while the event loop is
        # still open. Otherwise its __del__ runs after the loop has been
        # closed at interpreter shutdown and emits an ignored
        # "Event loop is closed" RuntimeError.
        transport = getattr(proc, "_transport", None)
        if transport is not None:
            try:
                transport.close()
            except Exception:
                pass
            try:
                proc._transport = None
            except Exception:
                pass

    async def run(self, command):
        if not self._started:
            raise ValueError("Session has not started.")
        if self._process.returncode is not None:
            self._started = False
            raise ValueError(
                f"Bash has exited with returncode {self._process.returncode}"
            )
        if self._timed_out:
            raise ValueError(
                f"Timed out: bash has not returned in {self._timeout} seconds and must be restarted."
            )

        # A unique sentinel per command means command output that happens to
        # contain a fixed marker string can no longer truncate the result.
        sentinel = f"__BASH_TOOL_DONE_{uuid.uuid4().hex}__"

        # Always put the sentinel echo on its own line. Appending it with
        # `; echo ...` on the same line (as the old implementation did) broke
        # commands ending with a heredoc delimiter: `EOF` became
        # `EOF; echo ...`, so the heredoc never terminated and the tool hung
        # until timeout.
        self._process.stdin.write(
            command.encode() + f"\necho {sentinel}\n".encode()
        )
        await self._process.stdin.drain()

        loop = asyncio.get_running_loop()
        start_time = loop.time()

        while True:
            if loop.time() - start_time > self._timeout:
                self._timed_out = True
                self.stop()
                raise ValueError(
                    f"Timed out: bash has not returned in {self._timeout} seconds and must be restarted."
                )

            await asyncio.sleep(self._output_delay)

            stdout_data = self._process.stdout._buffer.decode(errors="ignore")
            stderr_data = self._process.stderr._buffer.decode(errors="ignore")

            if sentinel in stdout_data:
                output = stdout_data[: stdout_data.index(sentinel)]
                break

            # If the shell died before echoing the sentinel (e.g. the command
            # was `exit`), fail fast instead of waiting out the timeout.
            if self._process.returncode is not None:
                self._started = False
                raise ValueError(
                    f"Bash has exited with returncode {self._process.returncode}"
                )

        self._process.stdout._buffer.clear()
        self._process.stderr._buffer.clear()

        return output.strip(), stderr_data.strip()


def filter_error(error):
    """Drop bash's own terminal-setup noise if it ever appears.

    The shell is non-interactive now, so this should be a no-op, but keep a
    small, bounds-safe filter for the stray ioctl warning that some commands
    still emit when run without a tty.
    """
    filtered_lines = []
    skip = 0
    for line in (error or "").splitlines():
        if "Inappropriate ioctl for device" in line:
            skip = 2
            continue
        if skip > 0:
            skip -= 1
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines).strip()


# The shell lives across tool calls. asyncio.run() would create a fresh event
# loop for each call and close it at the end, which would also tear down the
# subprocess transport and the shell with it -- so instead we keep one loop
# and run coroutines on it with run_until_complete, which is safe to call
# repeatedly from the same thread (the agent loop is single-threaded).
_LOOP = None
_SESSION = None
_LOCK = threading.Lock()


def _get_loop():
    global _LOOP
    if _LOOP is None or _LOOP.is_closed():
        _LOOP = asyncio.new_event_loop()
    return _LOOP


async def _start_session():
    session = BashSession()
    await session.start()
    return session


def _shutdown():
    global _SESSION
    try:
        if _SESSION is not None:
            _SESSION.stop()
    finally:
        _SESSION = None


atexit.register(_shutdown)


def tool_function(command):
    """Execute a command in the persistent bash shell."""
    global _SESSION
    with _LOCK:
        loop = _get_loop()
        try:
            if _SESSION is None or not _SESSION._started or _SESSION._timed_out:
                _SESSION = loop.run_until_complete(_start_session())
            output, error = loop.run_until_complete(_SESSION.run(command))
        except Exception as e:
            try:
                if _SESSION is not None:
                    _SESSION.stop()
            finally:
                _SESSION = None
            return f"Error: {str(e)}"

    error = filter_error(error)
    result = ""
    if output:
        result += output
    if error:
        result += "\nError:\n" + error
    return result.strip()


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python bash.py '<command>'")
    else:
        input_command = " ".join(sys.argv[1:])
        result = tool_function(input_command)
        print(result)
