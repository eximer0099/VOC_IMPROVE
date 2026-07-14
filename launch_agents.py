"""Launch and supervise all VOC gRPC agent services."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


AGENT_MODULES = (
    "agents.interpreter",
    "agents.retriever",
    "agents.summarizer",
    "agents.evaluator",
    "agents.critic",
    "agents.improver",
)


def stop_agents(processes: list[subprocess.Popen[bytes]]) -> None:
    """Ask every child to stop, then force-stop any that do not exit."""
    for process in processes:
        if process.poll() is None:
            process.terminate()

    deadline = time.monotonic() + 5
    for process in processes:
        remaining = max(0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> int:
    project_root = Path(__file__).resolve().parent
    processes: list[subprocess.Popen[bytes]] = []

    try:
        for module in AGENT_MODULES:
            process = subprocess.Popen(
                [sys.executable, "-m", module],
                cwd=project_root,
            )
            processes.append(process)
            print(f"Started {module} (PID {process.pid})", flush=True)

        while True:
            for module, process in zip(AGENT_MODULES, processes):
                return_code = process.poll()
                if return_code is not None:
                    print(
                        f"{module} exited unexpectedly with code {return_code}.",
                        file=sys.stderr,
                    )
                    return return_code or 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping all agents...", flush=True)
        return 0
    finally:
        stop_agents(processes)


if __name__ == "__main__":
    raise SystemExit(main())
