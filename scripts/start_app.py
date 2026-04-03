"""Start both the backend agent server and the frontend chat app."""

import os
import signal
import subprocess
import sys
import time


def main():
    """Launch the backend (FastAPI/uvicorn) and frontend (Next.js chat app)."""
    backend_port = os.getenv("BACKEND_PORT", "8000")
    frontend_port = os.getenv("CHAT_APP_PORT", "3000")

    # Start the backend agent server
    backend_cmd = [
        sys.executable, "-m", "uvicorn",
        "agent_server.start_server:app",
        "--host", "0.0.0.0",
        "--port", backend_port,
    ]

    print(f"Starting backend on port {backend_port}...")
    backend_proc = subprocess.Popen(backend_cmd)

    # Clone and start the frontend chat app
    frontend_dir = os.path.join(os.getcwd(), "e2e-chatbot-app-next")
    if not os.path.isdir(frontend_dir):
        print("Cloning frontend chat app...")
        subprocess.run(
            [
                "git", "clone", "--depth", "1",
                "https://github.com/databricks/app-templates.git",
                "--branch", "main",
                "--single-branch",
                "app-templates-tmp",
            ],
            check=True,
        )
        subprocess.run(
            ["mv", "app-templates-tmp/e2e-chatbot-app-next", frontend_dir],
            check=True,
        )
        subprocess.run(["rm", "-rf", "app-templates-tmp"], check=True)

    # Install frontend dependencies and start
    print(f"Starting frontend on port {frontend_port}...")
    frontend_env = os.environ.copy()
    frontend_env["PORT"] = frontend_port
    frontend_env["API_PROXY"] = os.getenv("API_PROXY", f"http://localhost:{backend_port}/invocations")

    frontend_proc = None
    if os.path.isfile(os.path.join(frontend_dir, "package.json")):
        subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
        frontend_proc = subprocess.Popen(
            ["npm", "run", "start"],
            cwd=frontend_dir,
            env=frontend_env,
        )

    def shutdown(signum, frame):
        backend_proc.terminate()
        if frontend_proc:
            frontend_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Wait for processes
    try:
        backend_proc.wait()
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()
