import argparse
import torch
import torch.nn as nn

from pathlib import Path
from neuraloperators.loading import load_deeponet_problem
from dataset.dataset import MeshData

def pred_problem(problem_file: Path, dataset_path: Path, branch_state_dict_path: Path,
                 trunk_state_dict_path: Path, save_file: Path) -> None:


    deeponet, _, _, dset, \
    _, _, _, _, mask_tensor = load_deeponet_problem(problem_file)

    x_data: MeshData = dset.x_data
    y_data: MeshData = dset.y_data

    evaluation_points = y_data.dof_coordinates[None,...].to(dtype=torch.get_default_dtype())

    from neuraloperators.deeponet import DeepONet
    class EvalWrapper(nn.Module):
        def __init__(self, deeponet: DeepONet, evaluation_points: torch.Tensor, mask_tensor: torch.Tensor):
            super().__init__()

            self.deeponet = deeponet

            if len(mask_tensor.shape) == 1:
                mask_tensor = mask_tensor[:,None]

            self.register_buffer("evaluation_points", evaluation_points)
            self.register_buffer("mask_tensor", mask_tensor)
            self.evaluation_points: torch.Tensor
            self.mask_tensor: torch.Tensor
            self.evaluation_points.requires_grad_(False)
            self.mask_tensor.requires_grad_(False)

            return
        
        def forward(self, uh: torch.Tensor) -> torch.Tensor:
            return uh + self.deeponet(uh, self.evaluation_points) * self.mask_tensor

    
    net = EvalWrapper(deeponet, evaluation_points, mask_tensor)
    deeponet.branch.load_state_dict(torch.load(branch_state_dict_path,map_location=torch.device("cpu")))
    deeponet.trunk.load_state_dict(torch.load(trunk_state_dict_path,map_location=torch.device("cpu")))

    # Change mask_tensor, evaluation_points, rebuild branch_encoder to account for change in dataset

    from dataset.dataset import load_MeshData, FEniCSDataset, ToDType
    x_data, y_data = load_MeshData(dataset_path)
    dset = FEniCSDataset(x_data, y_data, x_transform=ToDType(), y_transform=ToDType())

    mask_tensor = y_data.create_mask_function()
    evaluation_points = y_data.dof_coordinates[None,...].to(dtype=torch.get_default_dtype())

    from neuraloperators.encoders import SequentialEncoder, FlattenEncoder, InnerBoundaryFilterEncoder
    branch_encoder = SequentialEncoder(InnerBoundaryFilterEncoder(x_data), FlattenEncoder(-2))

    net.mask_tensor = mask_tensor # Make sure mask tensor is for correct dataset, not the one trained on, which is loaded in with the state-dict.
    net.evaluation_points = evaluation_points
    net.deeponet.branch_encoder = branch_encoder # Change to new branch encoder for correct dataset

    from tools.xdmf_io import pred_to_xdmf
    pred_to_xdmf(net, dset, save_file, overwrite=True)

    return


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-file", default=Path("hyperparameter_study/best_run/problem.json"), type=Path)
    parser.add_argument("--branch-state-dict", default=Path("hyperparameter_study/best_run/branch.pt"), type=Path)
    parser.add_argument("--trunk-state-dict", default=Path("hyperparameter_study/best_run/trunk.pt"), type=Path)
    parser.add_argument("--save-file", default=Path("output/fenics/best_run.pred.xdmf"), type=Path)
    parser.add_argument("--dataset", default=Path("dataset/learnext_period_p1"), type=Path)
    args = parser.parse_args()

    problem_file: Path = args.problem_file
    branch_state_dict: Path = args.branch_state_dict
    trunk_state_dict: Path = args.trunk_state_dict
    dataset_path: Path = args.dataset
    save_file: Path = args.save_file

    possible_collisions = [save_file.with_suffix(".xdmf"), save_file.with_suffix(".h5")]
    collisions = list(filter(lambda p: p.exists(), possible_collisions))

    if len(collisions) > 0:
        print(f"Save files {[str(p) for p in collisions]} already exist.")
        if input("Overwrite? (y/n): ").lower() != "y":
            quit()
        else:
            [p.unlink() for p in collisions]

    assert save_file.suffix == ".xdmf"
    pred_problem(problem_file, dataset_path, branch_state_dict, trunk_state_dict, save_file)

    return


if __name__ == "__main__":
    main()
