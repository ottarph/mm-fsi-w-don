import torch
import torch.nn as nn
from neuraloperators.loading import *

def test_model_builder():

    model_dict = {
        "MLP": {
            "widths": [2, 4, 2],
            "activation": "ReLU"
        }
    }

    model = build_model(model_dict)
    from neuraloperators.mlp import MLP
    assert isinstance(model, MLP)
    assert model.widths == [2, 4, 2]
    assert isinstance(model.activation, torch.nn.ReLU)

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
    test_model_builder()
    test_optimizer_builder()
    test_scheduler_builder()
    test_loss_fn_builder()
