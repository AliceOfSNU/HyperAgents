import os
import select
import signal
import subprocess
import threading
import time
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
    """A persistent, non-interactive bash shell shared across tool calls.

    The previous implementation created a fresh `bash -i` process for every
    tool call. That looked stateful but wasn't: `cd`, exported variables, and
    shell settings never survived to the next call, directly contradicting the
    tool description. A single long-lived `bash --norc --noprofile` process
    gives real persistence of working directory, environment variables, and
    background processes, and avoids interactive-shell prompt/ioctl noise on
    stderr.
    """

    def __init__(self):
        self._process = None
        self._timeout = 120.0  # seconds

    def start(self):
        if self._process is not None and self._process.poll() is None:
            return
        self._process = subprocess.Popen(
            ["/bin/bash", "--norc", "--noprofile"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
            preexec_fn=os.setsid,
        )

    def stop(self):
        if self._process is None:
            return
        if self._process.poll() is None:
            try:
                os.killpg(self._process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self._process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                self._process.wait(timeout=5)
        self._process = None

    def _read_available(self, stream):
        """Read whatever `stream` has buffered right now, without blocking."""
        import fcntl
        fd = stream.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        chunks = []
        try:
            while True:
                try:
                    data = os.read(fd, 65536)
                except BlockingIOError:
                    break
                if not data:
                    break
                chunks.append(data.decode(errors="ignore"))
        finally:
            fcntl.fcntl(fd, fcntl.F_SETFL, flags)
        return "".join(chunks)

    def run(self, command):
        if self._process is None or self._process.poll() is not None:
            raise ValueError("Bash has exited and must be restarted.")

        # A unique sentinel per command means legitimate command output that
        # contains the marker string can no longer truncate the result. The
        # old fixed `<<exit>>` sentinel did exactly that.
        sentinel = f"__HYPERAGENTS_BASH_SENTINEL_{uuid.uuid4().hex}__"
        try:
            # Write the sentinel echo as its own command line, never appended
            # to the user's last line with `;`. If the user command ends with
            # a heredoc delimiter (e.g. `cat > f <<'EOF'\n...\nEOF`), a `;`
            # would turn the delimiter line into `EOF; echo sentinel`, so the
            # heredoc would never terminate and every such call would hang
            # until timeout.
            self._process.stdin.write(f"{command}\necho '{sentinel}'\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            self.stop()
            raise ValueError(f"Bash has exited and must be restarted: {e}")

        stdout_data = ""
        stderr_data = ""
        start_time = time.time()

        while True:
            if time.time() - start_time > self._timeout:
                self.stop()
                raise ValueError(
                    f"Timed out: bash has not returned in {self._timeout} seconds and must be restarted."
                )

            ready, _, _ = select.select(
                [self._process.stdout, self._process.stderr], [], [], 0.1
            )

            for stream in ready:
                if stream is self._process.stdout:
                    stdout_data += self._read_available(stream)
                else:
                    stderr_data += self._read_available(stream)

            if sentinel in stdout_data:
                output = stdout_data[: stdout_data.index(sentinel)]
                # Discard anything that arrived after the sentinel for this
                # command, so it can't bleed into the next command's result.
                self._read_available(self._process.stdout)
                self._read_available(self._process.stderr)
                return output.strip(), stderr_data.strip()

            if self._process.poll() is not None:
                self.stop()
                raise ValueError(
                    f"Bash exited with return code {self._process.returncode}"
                )


def filter_error(error):
    """Remove the small amount of interactive-shell startup noise that could
    still appear if bash is ever launched in an environment that forces
    interactive mode."""
    filtered_lines = []
    for line in error.splitlines():
        if "Inappropriate ioctl for device" in line:
            continue
        if "no job control in this shell" in line:
            continue
        if "cannot set terminal process group" in line:
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines).strip()


_BASH_SESSION = None
_BASH_SESSION_LOCK = threading.Lock()


def _new_session():
    session = BashSession()
    session.start()
    return session


def tool_function_call(command):
    """Execute a command in the persistent bash shell."""
    global _BASH_SESSION
    with _BASH_SESSION_LOCK:
        try:
            if _BASH_SESSION is None:
                _BASH_SESSION = _new_session()

            try:
                output, error = _BASH_SESSION.run(command)
            except ValueError:
                # The persistent shell timed out or exited. Rebuild it and
                # retry once; if the command is what killed the shell, the
                # retry will fail and the error is returned below.
                _BASH_SESSION.stop()
                _BASH_SESSION = None
                _BASH_SESSION = _new_session()
                output, error = _BASH_SESSION.run(command)
        except Exception as e:
            return f"Error: {str(e)}"

    error = filter_error(error)
    result = ""
    if output:
        result += output
    if error:
        result += "\nError:\n" + error
    return result.strip()


def tool_function(command):
    return tool_function_call(command)


if __name__ == "__main__":
    # Example usage
    import sys

    # Check if the script is called with arguments
    if len(sys.argv) < 2:
        print("Usage: python bash.py '<command>'")
    else:
        # Extract the command from the command-line arguments
        input_command = ' '.join(sys.argv[1:])
        # Run the tool_function
        result = tool_function(input_command)
        print(result)
