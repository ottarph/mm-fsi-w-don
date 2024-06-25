import torch
import torch.nn as nn
from pathlib import Path
import json

import neuraloperators.networks
import neuraloperators.deeponet
import neuraloperators.encoders

import dataset.dataset

from typing import Literal
from os import PathLike
from torch.utils.data import DataLoader, random_split

class ModelBuilder:

    def Sequential(model_dicts: list[dict]) -> nn.Sequential:

        return nn.Sequential(*(build_model(model_dict)
                                for model_dict in model_dicts))
    
    def activation(activation_type: str) -> nn.Module:

        activations = {"ReLU": nn.ReLU(), 
                       "Tanh": nn.Tanh(),
                       "Sigmoid": nn.Sigmoid()}
        return activations[activation_type]

    def MLP(mlp_dict: dict) -> neuraloperators.networks.MLP:

        activation = ModelBuilder.activation(mlp_dict["activation"])
        widths = mlp_dict["widths"]
        return neuraloperators.networks.MLP(widths, activation)
    

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


from neuraloperators.cost_functions import RelativeMSELoss
class LossBuilder:

    def MSELoss(mseloss_dict: dict) -> nn.MSELoss:
        return nn.MSELoss(**mseloss_dict, reduction="mean")
    
    def L1Loss(l1loss_dict: dict) -> nn.L1Loss:
        return nn.L1Loss(**l1loss_dict, reduction="mean")
    
    def RelativeMSELoss(rel_mseloss_dict: dict) -> RelativeMSELoss:
        return RelativeMSELoss(**rel_mseloss_dict)
    
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
    
    def ExponentialLR(optimizer: torch.optim.Optimizer, explr_dict: dict) -> torch.optim.lr_scheduler.ExponentialLR:
        if "gamma" in explr_dict.keys():
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, **explr_dict)
        elif "lr_last" in explr_dict.keys() and "num_epochs" in explr_dict.keys():
            lr_0 = optimizer.param_groups[0]["lr"]
            lr_last = explr_dict["lr_last"]
            T = explr_dict["num_epochs"]
            from numpy import exp, log
            gamma = exp( ( log(lr_last) - log(lr_0) ) / T )
            explr_dict.pop("lr_last"); explr_dict.pop("num_epochs")
            explr_dict["gamma"] = gamma
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, **explr_dict)

LR_Scheduler = torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau
def build_scheduler(optimizer, scheduler_dict: dict) -> LR_Scheduler:

    assert len(scheduler_dict.keys()) == 1
    key = next(iter(scheduler_dict.keys()))
    val = scheduler_dict[key]

    scheduler: LR_Scheduler = getattr(SchedulerBuilder, key)(optimizer, val)

    return scheduler


class EncoderBuilder:

    
    def CoordinateInsertEncoder(mesh_data: dataset.dataset.MeshData, coord_insert_dict: dict) -> neuraloperators.encoders.CoordinateInsertEncoder:
        return neuraloperators.encoders.CoordinateInsertEncoder(mesh_data=mesh_data, **coord_insert_dict)
    
    def BoundaryFilterEncoder(mesh_data: dataset.dataset.MeshData, boundary_filter_dict: dict) -> neuraloperators.encoders.BoundaryFilterEncoder:
        return neuraloperators.encoders.BoundaryFilterEncoder(mesh_data=mesh_data, **boundary_filter_dict)

    def InnerBoundaryFilterEncoder(mesh_data: dataset.dataset.MeshData, boundary_filter_dict: dict) -> neuraloperators.encoders.InnerBoundaryFilterEncoder:
        return neuraloperators.encoders.InnerBoundaryFilterEncoder(mesh_data=mesh_data, **boundary_filter_dict)

    def FlattenEncoder(mesh_data: dataset.dataset.MeshData, flatten_dict: dict) -> neuraloperators.encoders.FlattenEncoder:
        return  neuraloperators.encoders.FlattenEncoder(**flatten_dict)

    def SequentialEncoder(mesh_data: dataset.dataset.MeshData, encoder_dicts: list[dict]) -> neuraloperators.encoders.SequentialEncoder:
        return neuraloperators.encoders.SequentialEncoder(*(build_encoder(mesh_data, encoder_dict) for encoder_dict in encoder_dicts))
    
    def IdentityEncoder(mesh_data: dataset.dataset.MeshData, ident_dict: dict) -> neuraloperators.encoders.IdentityEncoder:
        return neuraloperators.encoders.IdentityEncoder(**ident_dict)


def build_encoder(mesh_data: dataset.dataset.MeshData, encoder_dict: dict) -> neuraloperators.encoders.Encoder:

    assert len(encoder_dict.keys()) == 1
    key = next(iter(encoder_dict.keys()))
    val = encoder_dict[key]

    encoder: neuraloperators.encoders.Encoder = getattr(EncoderBuilder, key)(mesh_data, val)

    return encoder


def load_deeponet_problem(problemdict: PathLike | dict) \
    -> tuple[neuraloperators.deeponet.DeepONet, DataLoader, DataLoader, dataset.dataset.FEniCSDataset,
             torch.optim.Optimizer, LR_Scheduler, nn.modules.loss._Loss, 
             int, torch.Tensor]:
    
    if isinstance(problemdict, PathLike):
        import json
        problem_path = Path(problemdict)
        
        with open(problem_path, "r") as infile:
            problemdict = json.loads(infile.read())

    if "seed" in problemdict.keys():
        torch.manual_seed(problemdict["seed"])

    from dataset.dataset import load_MeshData, FEniCSDataset, ToDType
    from torch.utils.data import DataLoader
    x_data, y_data = load_MeshData(problemdict["dataset"]["directory"], problemdict["dataset"]["style"])
    dset = FEniCSDataset(x_data, y_data, 
                    x_transform=ToDType("default"),
                    y_transform=ToDType("default"))
    if "train_val_split" in problemdict["dataset"].keys():
        train_dataset, val_dataset = random_split(dset, problemdict["dataset"]["train_val_split"])
        train_dataloader = DataLoader(train_dataset, batch_size=problemdict["dataset"]["batch_size"])
        val_dataloader = DataLoader(val_dataset, batch_size=problemdict["dataset"]["batch_size"])
    else:
        train_dataset, val_dataset = dset, dset
        train_dataloader = DataLoader(train_dataset, batch_size=problemdict["dataset"]["batch_size"])
        val_dataloader = DataLoader(val_dataset, batch_size=problemdict["dataset"]["batch_size"])

    sensors = x_data.dof_coordinates

    branch_net = build_model(problemdict["branch"])
    trunk_net = build_model(problemdict["trunk"])

    branch_encoder = build_encoder(x_data, problemdict["branch_encoder"])
    trunk_encoder = build_encoder(y_data, problemdict["trunk_encoder"])

    if problemdict["final_bias"] == False:
        final_bias = None
    else:
        raise NotImplementedError

    from neuraloperators.deeponet import DeepONet
    deeponet = DeepONet(branch_encoder, branch_net, trunk_encoder, trunk_net, 
                        x_data.function_space.ufl_element().value_shape()[0], # This will probably fail if trying to use scalar function spaces.
                        y_data.function_space.ufl_element().value_shape()[0], # This will probably fail if trying to use scalar function spaces.
                        final_bias=final_bias, combine_style=problemdict["combine_style"],
                        sensors=sensors)

    optimizer = build_optimizer(deeponet.parameters(), problemdict["optimizer"])
    scheduler = build_scheduler(optimizer, problemdict["scheduler"])
    loss_fn = build_loss_fn(problemdict["loss_fn"])
    num_epochs = problemdict["num_epochs"]

    mask_tensor = y_data.create_mask_function(problemdict["mask_function_f"])

    return deeponet, train_dataloader, val_dataloader, dset, optimizer, scheduler, loss_fn, num_epochs, mask_tensor



if __name__ == "__main__":

    path = Path("tests/default_problem.json")

    deeponet, train_dataloader, val_dataloader, dset, optimizer, scheduler, loss_fn, num_epochs, mask_tensor = load_deeponet_problem(path)

    print(deeponet)
    uh, _ = next(iter(train_dataloader))
    print(f"{uh.shape = }")
    print(f"{deeponet.branch_encoder(uh).shape = }")
    y = dset.y_data.dof_coordinates[None,...].to(dtype=torch.get_default_dtype())
    print(f"{deeponet(uh, y).shape = }")
    print(f"{(uh + deeponet(uh, y) * mask_tensor).shape = }")

