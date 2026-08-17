import asyncio
import os

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
    """A session of a bash shell."""
    def __init__(self):
        self._started = False
        self._process = None
        self._timed_out = False
        self._timeout = 120.0  # seconds
        self._sentinel = "<<exit>>"
        self._output_delay = 0.2  # seconds

    async def start(self):
        if self._started:
            return
        self._process = await asyncio.create_subprocess_shell(
            "/bin/bash -i",
            preexec_fn=os.setsid,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy()  # Ensures inheritance of the current environment
        )
        self._started = True

    def stop(self):
        if not self._started:
            return
        if self._process.returncode is None:
            self._process.terminate()
        self._process = None
        self._started = False

    async def run(self, command):
        if not self._started:
            raise ValueError("Session has not started.")
        if self._process.returncode is not None:
            raise ValueError(f"Bash has exited with returncode {self._process.returncode}")
        if self._timed_out:
            raise ValueError(
                f"Timed out: bash has not returned in {self._timeout} seconds and must be restarted."
            )

        # Send command followed by sentinel echo
        self._process.stdin.write(
            command.encode() + f"; echo '{self._sentinel}'\n".encode()
        )
        await self._process.stdin.drain()

        output_lines = []
        error_lines = []
        start_time = asyncio.get_event_loop().time()

        # Read stdout until sentinel
        try:
            while True:
                if asyncio.get_event_loop().time() - start_time > self._timeout:
                    self._timed_out = True
                    raise ValueError(
                        f"Timed out: bash has not returned in {self._timeout} seconds and must be restarted."
                    )
                try:
                    line = await asyncio.wait_for(self._process.stdout.readline(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                if not line:
                    # EOF; stop
                    break
                text = line.decode(errors='ignore')
                if self._sentinel in text:
                    # Remove sentinel and any trailing text
                    idx = text.index(self._sentinel)
                    if idx > 0:
                        output_lines.append(text[:idx])
                    break
                output_lines.append(text)
        except Exception as e:
            self._timed_out = True
            raise ValueError(str(e))

        # Read any remaining stderr (non-blocking)
        try:
            while True:
                try:
                    line = await asyncio.wait_for(self._process.stderr.readline(), timeout=0.2)
                except asyncio.TimeoutError:
                    break
                if not line:
                    break
                error_lines.append(line.decode(errors='ignore'))
        except Exception:
            pass

        output = ''.join(output_lines).strip()
        error = ''.join(error_lines).strip()
        return output, error

def filter_error(error):
    # Filter out errors that we do not want to see
    filtered_lines = []
    i = 0
    error_lines = error.splitlines()
    while i < len(error_lines):
        line = error_lines[i]

        # Skip the next lines if ioctl error, add relevant lines
        if "Inappropriate ioctl for device" in line:
            i += 3
            if '<<exit>>' in error_lines[i]:
                i += 1
            while i < len(error_lines) - 1:
                filtered_lines.append(error_lines[i])
                i += 1
            i += 1
            continue

        filtered_lines.append(line)
        i += 1
    return '\n'.join(filtered_lines).strip()

# Global session so state persists across separate tool calls, matching the
# tool description's promise: "State is persistent across command calls".
_global_bash_session = None

def get_bash_session():
    global _global_bash_session
    if _global_bash_session is None:
        _global_bash_session = BashSession()
    return _global_bash_session

async def tool_function_call(command):
    """Execute a command in the bash shell."""
    try:
        bash_session = get_bash_session()

        if not bash_session._started:
            await bash_session.start()

        output, error = await bash_session.run(command)
        error = filter_error(error)
        result = ""
        if output:
            result += output
        if error:
            result += "\nError:\n" + error
        return result.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def tool_function(command):
    return asyncio.run(tool_function_call(command))

if __name__ == "__main__":
    # Example usage
    import sys

    # Check if the script is called with arguments
    if len(sys.argv) < 2:
        print("Usage: python bash.py '<command>'")
    else:
        # Extract the command from the command-line arguments
        input_command = ' '.join(sys.argv[1:])
        # Run the tool_function asynchronously
        result = tool_function(input_command)
        print(result)
