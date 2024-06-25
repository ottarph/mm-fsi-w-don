import subprocess as sp
import shutil
from pathlib import Path
import os

def main():
    
    env = os.environ.copy()
    if "PYTHONPATH" in env.keys():
        env["PYTHONPATH"] = env["PYTHONPATH"] + ":" + "."
    else:
        env["PYTHONPATH"] = "."

    sp.run(["python3", "scripts/download_data.py"], env=env)

    Path("output/data").mkdir(parents=True, exist_ok=True)
    Path("output/figures").mkdir(parents=True, exist_ok=True)

    sp.run(["python3", "scripts/compute_biharm_mesh_data.py", "dataset/learnext_period_p1", "output/data/biharm_min_mq.csv"], env=env)

    sp.run(["python3", "hyperparameter_study/study.py"], env=env)

    shutil.copy("hyperparameter_study/best_run/problem.json", "random_initialization/base_problem.json")

    sp.run(["python3", "random_initialization/random_initialization_study.py"], env=env)
    sp.run(["python3", "random_initialization/make_plots.py"], env=env)

    sp.run(["python3", "grav_test/run_grav_test.py"], env=env)

    return

if __name__ == "__main__":
    main()
