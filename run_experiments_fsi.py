import subprocess as sp
from pathlib import Path
import os

def main():
    
    env = os.environ.copy()
    if "PYTHONPATH" in env.keys():
        env["PYTHONPATH"] = env["PYTHONPATH"] + ":" + "."
    else:
        env["PYTHONPATH"] = "."


    Path("output/figures").mkdir(parents=True, exist_ok=True)

    sp.run(["python3", "deeponet_extension/run.py"], env=env)
    sp.run(["python3", "deeponet_extension/run_biharm.py"], env=env)
    
    sp.run(["python3", "deeponet_extension/plot_quantities.py"], env=env)

    return

if __name__ == "__main__":
    main()
