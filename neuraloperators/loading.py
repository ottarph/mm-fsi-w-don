import torch
import torch.nn as nn
from pathlib import Path
import json

import neuraloperators.mlp
import neuraloperators.deeponet
from typing import Literal
from os import PathLike
from torch.utils.data import DataLoader

class ModelBuilder:

    def activation(activation_type: str) -> nn.Module:
        activations = {"ReLU": nn.ReLU(), 
                       "Tanh": nn.Tanh(),
                       "Sigmoid": nn.Sigmoid()}
        return activations[activation_type]

    def MLP(mlp_dict: dict) -> neuraloperators.mlp.MLP:

        activation = ModelBuilder.activation(mlp_dict["activation"])
        widths = mlp_dict["widths"]

        return neuraloperators.mlp.MLP(widths, activation)
    
    def Sequential(model_dicts: list[dict]) -> nn.Sequential:

        return nn.Sequential(*(build_model(model_dict)
                                for model_dict in model_dicts))
    
    def DeepONet(deeponet_dict: dict) -> neuraloperators.deeponet.DeepONet:

        raise NotImplementedError()



def build_model(model_dict: dict) -> nn.Module:

    assert len(model_dict.keys()) == 1
    key = next(iter(model_dict.keys()))
    val = model_dict[key]

    model: nn.Module = getattr(ModelBuilder, key)(val)

    return model


def load_model(model_dir: PathLike, load_state_dict: bool = True,
                            mode: Literal["json"] = "json") -> nn.Module:
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise ValueError("Non-existent directory.")

    if mode == "json":
        with open(model_dir / "model.json", "r") as infile:
            model_dict= json.loads(infile.read())
    else:
        raise ValueError
    model = build_model(model_dict)

    if load_state_dict:
        model.load_state_dict(torch.load(model_dir / "state_dict.pt"))

    return model

def load_deeponet_problem(problem_path: PathLike, mode: Literal["json"] = "json") -> tuple[neuraloperators.deeponet.DeepONet,
                                                                                          DataLoader]:
    
    import json
    problem_path = Path(problem_path)
    
    with open(problem_path, "r") as infile:
        problemdict = json.loads(infile.read())

    # print(problemdict)
    # print(problemdict["branch"])
    # print(problemdict["trunk"])
    # print(problemdict["dataset"])

    from dataset.dataset import load_MeshData, FEniCSDataset, OnBoundary
    from torch.utils.data import DataLoader
    x_data, y_data = load_MeshData(problemdict["dataset"]["directory"], problemdict["dataset"]["style"])
    dataset = FEniCSDataset(x_data, y_data, x_transform=OnBoundary(x_data))
    dataloader = DataLoader(dataset, batch_size=problemdict["dataset"]["batch_size"])
    x0, y0 = next(iter(dataloader))
    # print(x0.shape)
    # print(y0.shape)

    sensors = x_data.boundary_dof_coordinates
    # print(sensors.shape)

    from neuraloperators.loading import build_model

    branch_net = build_model(problemdict["branch"])
    trunk_net = build_model(problemdict["trunk"])

    from neuraloperators.deeponet import BranchNetwork, TrunkNetwork, DeepONet
    branch = BranchNetwork(branch_net, sensors, 2, 2, 2, 2, problemdict["width"])
    trunk = TrunkNetwork(trunk_net, 2, 2, 2, 2, problemdict["width"])

    # print(branch)
    # print(trunk)

    from neuraloperators.deeponet import DeepONet
    deeponet = DeepONet(branch, trunk, sensors, problemdict["final_bias"])

    return deeponet, dataloader

    # print(deeponet)

