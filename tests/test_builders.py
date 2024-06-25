import torch
import torch.nn as nn
from neuraloperators.loading import *

def test_mlp_builder():

    model_dict = {
        "MLP": {
            "widths": [2, 4, 2],
            "activation": "ReLU"
        }
    }

    model = build_model(model_dict)
    from neuraloperators.networks import MLP
    assert isinstance(model, MLP)
    assert model.widths == [2, 4, 2]
    assert isinstance(model.activation, torch.nn.ReLU)

    return

def test_sequential_builder():

    model_dict = {
        "Sequential": [
            {"MLP": {"widths": [2, 4, 6], "activation": "ReLU"}},
            {"MLP": {"widths": [6, 5, 3], "activation": "Tanh"}}
        ]
    }

    model = build_model(model_dict)
    x = torch.rand((20, 2))
    assert model(x).shape == (20, 3)
    assert isinstance(model[0], neuraloperators.networks.MLP)
    assert isinstance(model[1], neuraloperators.networks.MLP)

    return


def test_encoder_builder():
    from dataset.dataset import load_MeshData, FEniCSDataset, ToDType
    x_data, y_data = load_MeshData("dataset/learnext_period_p1", "folders")
    dataset = FEniCSDataset(x_data, y_data, 
                    x_transform=ToDType("default"),
                    y_transform=ToDType("default"))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    x0, _ = next(iter(dataloader))

    encoder_dict = {
        "SequentialEncoder": [
            {"IdentityEncoder": {}},
            {"CoordinateInsertEncoder": {}},
            {"BoundaryFilterEncoder": {}},
            {"FlattenEncoder": {"start_dim": -2}}
        ]
    }
    encoder = build_encoder(x_data, encoder_dict)
    boundar_filter_encoder = build_encoder(x_data, {"BoundaryFilterEncoder": {}})
    assert encoder(x0).shape == (x0.shape[0], (2+2)*boundar_filter_encoder.filter_tensor.shape[0])

    return

def test_optimizer_builder():

    adam_dict = {"lr": 1e-3, "weight_decay": 1e-4}
    opt_dict = {"Adam": adam_dict}

    model_dict = {"MLP": {"widths": [2, 4, 2], "activation": "ReLU"}}
    model = build_model(model_dict)

    optimizer = build_optimizer(model.parameters(), opt_dict)
    assert type(optimizer) == torch.optim.Adam
    assert optimizer.param_groups[0]["lr"] == 1e-3
    assert optimizer.param_groups[0]["weight_decay"] == 1e-4

    return

def test_scheduler_builder():

    sched_dict = {"ConstantLR": {"factor": 0.5, "total_iters": 4}}

    model_dict = {"MLP": {"widths": [2, 4, 2], "activation": "ReLU"}}
    model = build_model(model_dict)
    opt_dict = {"Adam": {"lr": 1e-3, "weight_decay": 1e-4}}
    optimizer = build_optimizer(model.parameters(), opt_dict)
    
    scheduler = build_scheduler(optimizer, sched_dict)

    assert scheduler.factor == 0.5
    assert scheduler.total_iters == 4

    return

def test_loss_fn_builder():

    loss_dict = {"MSELoss": {}}

    loss = build_loss_fn(loss_dict)

    assert isinstance(loss, nn.MSELoss)

    return

if __name__ == "__main__":
    test_mlp_builder()
    test_sequential_builder()
    test_encoder_builder()
    test_optimizer_builder()
    test_scheduler_builder()
    test_loss_fn_builder()
