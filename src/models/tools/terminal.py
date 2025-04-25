from models.tools import Tool
import docker
from pydantic import BaseModel


class Terminal:
    def __init__(self, image="ubuntu:latest"):
        self.client = docker.from_env()
        self.container = self.client.containers.run(
            image=image, command="/bin/bash", detach=True, tty=True
        )

    def run_command(self, command: str) -> tuple[int, str]:
        exit_code, output = self.container.exec_run(f'bash -c "{command}"')
        return exit_code, output.decode("utf-8", errors="replace")

    def write_file(self, path: str, content: str) -> tuple[int, str]:
        # Create a properly escaped version of the content
        # This method uses base64 encoding to avoid escaping issues
        import base64

        encoded_content = base64.b64encode(content.encode("utf-8")).decode("ascii")
        command = f"echo '{encoded_content}' | base64 -d > {path}"
        return self.run_command(command)

    def read_file(self, path: str) -> str:
        exit_code, output = self.run_command(f"cat {path} 2>/dev/null")
        if exit_code != 0:
            return ""
        return output

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.container.stop()
        self.container.remove()


def _run_terminal(command: str) -> str:
    with Terminal() as env:
        exit_code, output = env.run_command(command)
        if exit_code != 0:
            return f"Command failed with exit code {exit_code}: {output}"
        return output


class TerminalArgs(BaseModel):
    command: str


terminal_tool = Tool(
    name="terminal",
    description="Execute a command in a terminal and return the output",
    function=_run_terminal,
    argument_schema=TerminalArgs,
)


# Example usage
if __name__ == "__main__":
    with Terminal() as env:
        print("$ mkdir -p /tmp\n", env.run_command("mkdir -p /tmp")[1])

        # Change directory command won't persist across exec calls
        # Each command runs in its own context
        print("$ ls -la /tmp\n", env.run_command("ls -la /tmp")[1])

        print("$ writing to /tmp/hello.txt")
        exit_code, output = env.write_file("/tmp/hello.txt", "Hello from Docker!")
        print(f"Exit code: {exit_code}, Output: {output}")

        # Verify the file exists
        print("$ ls -la /tmp/hello.txt\n", env.run_command("ls -la /tmp/hello.txt")[1])

        print("$ reading /tmp/hello.txt")
        content = env.read_file("/tmp/hello.txt")
        print(f"Content: {content}")

        special_content = "Line 1\nLine 2\nSpecial chars: !@#$%^&*()_+{}|:<>?'\""
        print("\n$ writing special characters to /tmp/special.txt")
        env.write_file("/tmp/special.txt", special_content)

        print("$ reading /tmp/special.txt")
        special_result = env.read_file("/tmp/special.txt")
        print(f"Content: {special_result}")
        print(f"Matches original: {special_content == special_result}")

        print("\n$ ls /tmp\n", env.run_command("ls -la /tmp")[1])
