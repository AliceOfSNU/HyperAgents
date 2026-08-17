import asyncio
import os
import subprocess

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
    """A simple synchronous bash session with a persistent working directory and timeout."""
    TIMEOUT = 180.0  # seconds

    def __init__(self):
        self.cwd = os.getcwd()

    def run(self, command):
        stripped = command.strip()

        # Handle a leading `cd` so the working directory persists across tool calls.
        if stripped.startswith("cd "):
            parts = stripped.split("&&", 1)
            cd_target = parts[0][3:].strip().strip('"').strip("'")
            new_cwd = (
                cd_target
                if os.path.isabs(cd_target)
                else os.path.normpath(os.path.join(self.cwd, cd_target))
            )
            if not os.path.isdir(new_cwd):
                return f"Error: directory '{cd_target}' does not exist"
            self.cwd = new_cwd
            command = parts[1].strip() if len(parts) > 1 else "pwd"

        # Always run in the persisted working directory.
        full_command = f"cd {self.cwd} && {command}" if command else f"cd {self.cwd} && pwd"
        try:
            result = subprocess.run(
                full_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT,
            )
            output = result.stdout.strip()
            error = result.stderr.strip()
            if result.returncode != 0 and error:
                output = output + "\nError:\n" + error if output else "Error:\n" + error
            return output
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {self.TIMEOUT} seconds. Background long-running tasks with e.g. 'nohup ... &' and then poll for completion."

# Module-level session so working directory persists across tool calls.
_GLOBAL_BASH_SESSION = BashSession()

def tool_function(command):
    try:
        return _GLOBAL_BASH_SESSION.run(command)
    except Exception as e:
        return f"Error: {str(e)}"

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
