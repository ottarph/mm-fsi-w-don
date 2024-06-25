import dolfin as df
import torch
import torch.nn as nn
from pathlib import Path
import json

import deeponet_extension.networks
import deeponet_extension.deeponet
import deeponet_extension.encoders


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

    def MLP(mlp_dict: dict) -> deeponet_extension.networks.MLP:

        activation = ModelBuilder.activation(mlp_dict["activation"])
        widths = mlp_dict["widths"]
        return deeponet_extension.networks.MLP(widths, activation)
    

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




class EncoderBuilder:


    def ExtractBoundaryDofEncoder(fspace: df.FunctionSpace, extract_bdof_dict: dict) -> deeponet_extension.encoders.ExtractBoundaryDofEncoder:
        return deeponet_extension.encoders.ExtractBoundaryDofEncoder(fspace=fspace, **extract_bdof_dict)
    
    def RandomSelectEncoder(fspace: df.FunctionSpace, rand_select_dict: dict) -> deeponet_extension.encoders.RandomSelectEncoder:
        return deeponet_extension.encoders.RandomSelectEncoder(**rand_select_dict)
    
    def RandomPermuteEncoder(fspace: df.FunctionSpace, rand_perm_dict: dict) -> deeponet_extension.encoders.RandomPermuteEncoder:
        return deeponet_extension.encoders.RandomPermuteEncoder(**rand_perm_dict)

    def FlattenEncoder(fspace: df.FunctionSpace, flatten_dict: dict) -> deeponet_extension.encoders.FlattenEncoder:
        return  deeponet_extension.encoders.FlattenEncoder(**flatten_dict)

    def SequentialEncoder(fspace: df.FunctionSpace, encoder_dicts: list[dict]) -> deeponet_extension.encoders.SequentialEncoder:
        return deeponet_extension.encoders.SequentialEncoder(*(build_encoder(fspace, encoder_dict) for encoder_dict in encoder_dicts))

    def IdentityEncoder(fspace: df.FunctionSpace, ident_dict: dict) -> deeponet_extension.encoders.IdentityEncoder:
        return deeponet_extension.encoders.IdentityEncoder(**ident_dict)


def build_encoder(fspace: df.FunctionSpace, encoder_dict: dict) -> deeponet_extension.encoders.Encoder:

    assert len(encoder_dict.keys()) == 1
    key = next(iter(encoder_dict.keys()))
    val = encoder_dict[key]

    encoder: deeponet_extension.encoders.Encoder = getattr(EncoderBuilder, key)(fspace, val)

    return encoder


def load_deeponet_problem(problemdict: PathLike | dict, fspace: df.Function,
                          state_dict_path: PathLike | None = None) \
    -> tuple[deeponet_extension.deeponet.DeepONet, str]:
    
    if isinstance(problemdict, PathLike):
        import json
        problem_path = Path(problemdict)
        
        with open(problem_path, "r") as infile:
            problemdict = json.loads(infile.read())

    if "seed" in problemdict.keys():
        torch.manual_seed(problemdict["seed"])

    branch_net = build_model(problemdict["branch"])
    trunk_net = build_model(problemdict["trunk"])

    branch_encoder = build_encoder(fspace, problemdict["branch_encoder"])
    trunk_encoder = build_encoder(fspace, problemdict["trunk_encoder"])

    if problemdict["final_bias"] == False:
        final_bias = None
    else:
        raise NotImplementedError

    mask_function_f = problemdict["mask_function_f"]

    from deeponet_extension.deeponet import DeepONet
    deeponet = DeepONet(branch_encoder, branch_net, trunk_encoder, trunk_net, 
                        2, 2, final_bias=final_bias, combine_style=problemdict["combine_style"])
    
    if state_dict_path is not None:
        state_dict_path = Path(state_dict_path)
        deeponet.branch.load_state_dict(torch.load(state_dict_path / "branch.pt", map_location="cpu"))
        deeponet.trunk.load_state_dict(torch.load(state_dict_path / "trunk.pt", map_location="cpu"))


    return deeponet, mask_function_f



if __name__ == "__main__":

    msh_path = Path("Output/Extension/learnext_period_p2/output.xdmf")
    msh_file = df.XDMFFile(str(msh_path))

    msh = df.Mesh()
    msh_file.read(msh)

    fspace = df.VectorFunctionSpace(msh, "CG", 1)


    problem_dir = Path("deeponet_extension/models/best_run")
    problem_path = problem_dir / "problem.json"

    deeponet, mask_function_f = load_deeponet_problem(problem_path, fspace, problem_dir)

    print(deeponet)
    print(mask_function_f)
    


