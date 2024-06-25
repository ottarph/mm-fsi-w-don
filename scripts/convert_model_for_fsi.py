import json
from pathlib import Path
import copy
import shutil

def main():

    fsi_model_dir = Path("deeponet_extension/models/best_run")
    fsi_model_dir.mkdir(parents=True, exist_ok=True)

    hp_best_run_dir = Path("hyperparameter_study/best_run")
    hp_problem_file = hp_best_run_dir / "problem.json"

    with open(hp_problem_file, "r") as infile:
        hp_problem_dict = json.load(infile)

    shutil.copy(hp_best_run_dir / "branch.pt", fsi_model_dir / "branch.pt")
    shutil.copy(hp_best_run_dir / "trunk.pt", fsi_model_dir / "trunk.pt")

    fsi_problem_dict = copy.deepcopy(hp_problem_dict)
    
    fsi_problem_dict["branch_encoder"] = {
        "SequentialEncoder": [
            {
                "ExtractBoundaryDofEncoder": {
                    "dof_coords_file_path": "output/data/sensor_dof_coords.cg1.txt"
                }
            },
            {
                "FlattenEncoder": {
                    "start_dim": -2
                }
            }
        ]
    }

    with open(fsi_model_dir / "problem.json", "w") as outfile:
        json.dump(fsi_problem_dict, outfile, indent=4)


    return

if __name__ == "__main__":
    main()
