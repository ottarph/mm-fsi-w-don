import dolfin as df
import torch
import torch.nn as nn

import json

from pathlib import Path

from neuraloperators.loading import load_deeponet_problem


def param_count(model: nn.Module) -> int:

    return sum(map(torch.numel, model.parameters()))



if __name__ == "__main__":
    df.set_log_active(False)

    CONFIG_PATH = "hyperparameter_study/study_config.json"
    BASE_PROBLEM_FILE_PATH = "hyperparameter_study/base_problem.json"

    with open(CONFIG_PATH, "r") as infile:
        config_dict = json.load(infile)

    with open(BASE_PROBLEM_FILE_PATH, "r") as infile:
        base_problem_dict = json.load(infile)

    print()

    widths = config_dict["widths"]
    depths = config_dict["depths"]
    basis_sizes = config_dict["basis_sizes"]

    
    for d in depths:
        for w in widths:
            for p in basis_sizes:
                problem_dict = {**base_problem_dict}
                problem_dict["branch"]["MLP"]["widths"] = [412] + [w] * d + [p]
                problem_dict["trunk"]["MLP"]["widths"] = [2] + [w] * d + [p]

                deeponet, *_ = load_deeponet_problem(problem_dict)
                print(f"{d = }, {w = }, {p = }: {param_count(deeponet) = :7d} ({param_count(deeponet):.2e})")

