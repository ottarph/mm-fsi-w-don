
from scripts.run_deeponet_training import run_boundary_problem

from pathlib import Path
import json

def test_run_boundary_problem(tmp_path):

    default_problem_path = Path("tests/default_problem.json")
    with open(default_problem_path, "r") as infile:
        default_probdict = json.loads(infile.read())
    
    default_probdict["num_epochs"] = 1

    tmp_prob_dict_path = tmp_path / "prob_dict.json"
    with open(tmp_prob_dict_path, "w") as outfile:
        json.dump(default_probdict, outfile)

    tmp_results_dir = tmp_path / "results"
    tmp_results_dir.mkdir()
    latest_results_dir = tmp_path / "latest"
    latest_results_dir.mkdir()

    run_boundary_problem(tmp_prob_dict_path, tmp_results_dir, save_xdmf=False, latest_results_dir=latest_results_dir)

    (tmp_results_dir / "branch.pt").unlink()
    (tmp_results_dir / "trunk.pt").unlink()
    (tmp_results_dir / "train_val_lr_hist.pdf").unlink()

    (latest_results_dir / "branch.pt").unlink()
    (latest_results_dir / "trunk.pt").unlink()
    (latest_results_dir / "train_val_lr_hist.pdf").unlink()

    valstr = (tmp_results_dir / "data/val.txt").read_text()
    assert len(valstr.split("\n")) == 2
    valstr = (latest_results_dir / "data/val.txt").read_text()
    assert len(valstr.split("\n")) == 2

    return
