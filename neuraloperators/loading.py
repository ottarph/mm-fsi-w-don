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


class LossBuilder:

    def MSELoss(mseloss_dict: dict) -> nn.MSELoss:
        return nn.MSELoss(**mseloss_dict)
    
    def L1Loss(l1loss_dict: dict) -> nn.L1Loss:
        return nn.L1Loss(**l1loss_dict)
    
def build_loss_fn(loss_dict: dict) -> nn.modules.loss._Loss:

    assert len(loss_dict.keys()) == 1
    key = next(iter(loss_dict.keys()))
    val = loss_dict[key]

    loss: nn.modules.loss._Loss = getattr(LossBuilder, key)(val)

    return loss


from typing import Generator
class OptimizerBuilder:
    def Adam(params: Generator, adam_dict: dict) -> torch.optim.Adam:
        return torch.optim.Adam(params, **adam_dict)
    
    def SGD(params: Generator, sgd_dict: dict) -> torch.optim.SGD:
        return torch.optim.SGD(params, **sgd_dict)
    
    def LBFGS(params: Generator, lbfgs_dict: dict) -> torch.optim.LBFGS:
        return torch.optim.LBFGS(params, **lbfgs_dict)
    
def build_optimizer(params: Generator, optimizer_dict: dict) -> torch.optim.Optimizer:

    assert len(optimizer_dict.keys()) == 1
    key = next(iter(optimizer_dict.keys()))
    val = optimizer_dict[key]

    optimizer: torch.optim.Optimizer = getattr(OptimizerBuilder, key)(params, val)

    return optimizer


class SchedulerBuilder:

    def ConstantLR(optimizer: torch.optim.Optimizer, constlr_dict: dict) -> torch.optim.lr_scheduler.ConstantLR:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, **constlr_dict)
    
    def ReduceLROnPlateau(optimizer: torch.optim.Optimizer, reducelr_dict: dict) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **reducelr_dict)

LR_Scheduler = torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau
def build_scheduler(optimizer, scheduler_dict: dict) -> LR_Scheduler:

    assert len(scheduler_dict.keys()) == 1
    key = next(iter(scheduler_dict.keys()))
    val = scheduler_dict[key]

    scheduler: LR_Scheduler = getattr(SchedulerBuilder, key)(optimizer, val)

    return scheduler


def load_deeponet_problem(problem_path: PathLike, mode: Literal["json"] = "json") \
    -> tuple[neuraloperators.deeponet.DeepONet, DataLoader, 
             torch.optim.Optimizer, LR_Scheduler, nn.modules.loss._Loss]:
    
    import json
    problem_path = Path(problem_path)
    
    with open(problem_path, "r") as infile:
        problemdict = json.loads(infile.read())

    from dataset.dataset import load_MeshData, FEniCSDataset, OnBoundary, ToDType
    from torch.utils.data import DataLoader
    x_data, y_data = load_MeshData(problemdict["dataset"]["directory"], problemdict["dataset"]["style"])
    dataset = FEniCSDataset(x_data, y_data, 
                    x_transform=nn.Sequential(OnBoundary(x_data), ToDType("default")),
                    y_transform=ToDType("default"))
    dataloader = DataLoader(dataset, batch_size=problemdict["dataset"]["batch_size"], shuffle=False)

    sensors = x_data.boundary_dof_coordinates

    from neuraloperators.loading import build_model

    branch_net = build_model(problemdict["branch"])
    trunk_net = build_model(problemdict["trunk"])

    from neuraloperators.deeponet import BranchNetwork, TrunkNetwork, DeepONet
    branch = BranchNetwork(branch_net, sensors, 2, 2, 2, 2, problemdict["width"])
    trunk = TrunkNetwork(trunk_net, 2, 2, 2, 2, problemdict["width"])


    from neuraloperators.deeponet import DeepONet
    deeponet = DeepONet(branch, trunk, sensors, problemdict["final_bias"])

    optimizer = build_optimizer(deeponet.parameters(), problemdict["optimizer"])
    scheduler = build_scheduler(optimizer, problemdict["scheduler"])
    loss_fn = build_loss_fn(problemdict["loss_fn"])

    return deeponet, dataloader, optimizer, scheduler, loss_fn

