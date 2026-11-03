"""python -m arhtmodelviewer serve [port] [--no-browser] [--lan]

--lan lets other computers poll the radars (GR2Analyst), not use the page.

The server runs as a child process under a small supervisor.  A GPU fault
(CUDA "illegal memory access" and the like) poisons the whole process -- every
later GPU call fails the same way -- so the child exits with RESTART_CODE and
the supervisor starts a fresh one on the same port."""
import os
import subprocess
import sys
import threading
import time

RESTART_CODE = 75


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "serve":
        argv = argv[1:]
    port = 8830
    nums = [a for a in argv if a.isdigit()]
    if nums:
        port = int(nums[0])
    open_browser = "--no-browser" not in argv
    lan = "--lan" in argv
    if "--child" in argv or "--no-supervisor" in argv:
        from .server import serve
        serve(port, open_browser=open_browser and "--child" not in argv, lan=lan)
        return
    supervise(port, open_browser, lan)


def supervise(port: int, open_browser: bool, lan: bool = False) -> None:
    cmd = [sys.executable, "-m", "arhtmodelviewer", "serve", str(port), "--no-browser", "--child"] + (["--lan"] if lan else [])
    env = dict(os.environ, ARHTMODELVIEWER_SUPERVISED="1", ARHTMODELVIEWER_PARENT=str(os.getpid()))
    first = True
    while True:
        if first and open_browser:
            import webbrowser
            threading.Timer(2.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}/")).start()
        first = False
        proc = subprocess.Popen(cmd, env=env)
        try:
            code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            return
        if code != RESTART_CODE:
            sys.exit(code)
        print("The GPU engine hit an error; restarting Atmos-WRF (your browser page reconnects by itself)...",
              flush=True)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
