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

def test_deepsets_builder():

    model_dict = {
        "DeepSets": {
            "representer": {
                "MLP": {"widths": [4, 8, 12], "activation": "ReLU"}
            },
            "processor": {
                "MLP": {"widths": [12, 16], "activation": "ReLU"}
            },
            "reduction": "mean"
        }
    }

    deepsets = build_model(model_dict)
    x = torch.rand((20, 10, 4))
    assert deepsets(x).shape == (20, 16)

    return

def test_split_additive_builder():

    split_add_dict = {
        "model_1": {"MLP": {"widths": [2, 64], "activation": "ReLU"}},
        "model_2": {"MLP": {"widths": [2, 64], "activation": "ReLU"}},
        "length_1": 2,
        "length_2": 2
    }

    split_add_model = build_model({"SplitAdditive": split_add_dict})
    x = torch.rand((5, 20, 4))
    assert split_add_model(x).shape == (5, 20, 64)

    return

def test_vidon_mha_builder():
    
    x = torch.rand((2, 60, 64))

    vmha_dict = {
        "d_enc": 64, "out_size": 32, "num_heads": 4,
        "weight_hidden_size": 256, "weight_num_layers": 4,
        "value_hidden_size": 256, "value_num_layers": 4
    }
    vmha = build_model({"VIDONMHA": vmha_dict})
    assert vmha(x).shape == (x.shape[0], 32*4)

    vmha_dict = {
        "d_enc": 64, "out_size": 32, "num_heads": 4,
        "weight_hidden_size": 256, "weight_num_layers": 4,
        "value_hidden_size": 256, "value_num_layers": 4,
        "weight_activation": "ReLU", "value_activation": "Tanh"
    }
    vmha = build_model({"VIDONMHA": vmha_dict})
    assert vmha(x).shape == (x.shape[0], 32*4)

    return

def test_vidon_builder():
    
    x = torch.rand((2, 60, 4))

    split_add_dict = {
        "model_1": {"MLP": {"widths": [2, 64], "activation": "ReLU"}},
        "model_2": {"MLP": {"widths": [2, 64], "activation": "ReLU"}},
        "length_1": 2,
        "length_2": 2
    }
    mha_dict = {
        "d_enc": 64, "out_size": 32, "num_heads": 4,
        "weight_hidden_size": 256, "weight_num_layers": 4,
        "value_hidden_size": 256, "value_num_layers": 4
    }
    processor_dict = {
        "MLP": {"widths": [32*4, 256, 32], "activation": "ReLU"}
    }
    vidon_dict = {
        "SplitAdditive": split_add_dict,
        "MultiHeadAttention": mha_dict,
        "Processor": processor_dict
    }

    vidon = build_model({"VIDON": vidon_dict})
    
    assert vidon(x).shape == (x.shape[0], 32)

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
            {"RandomPermuteEncoder": {"dim": -2, "unit_shape_length": 2}},
            {"RandomSelectEncoder": {"dim": -2, "unit_shape_length": 2, "num_inds": 6}},
            {"FlattenEncoder": {"start_dim": -2}}
        ]
    }
    encoder = build_encoder(x_data, encoder_dict)
    assert encoder(x0).shape == (x0.shape[0], (2+2)*6)

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
    test_deepsets_builder()
    test_split_additive_builder()
    test_vidon_mha_builder()
    test_vidon_builder()
    test_encoder_builder()
    test_optimizer_builder()
    test_scheduler_builder()
    test_loss_fn_builder()
