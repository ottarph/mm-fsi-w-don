
from neuraloperators.deeponet2 import *
from neuraloperators.mlp import MLP
from neuraloperators.encoders import FlattenEncoder, IdentityEncoder


def test_combine2():

    """
        Omega = [0,1]x[0,1]
        v: (v1, v2) two-dimensional vector field over Omega
        u: (u1, u2) two-dimensional vector field over Omega
    """

    bs = 6
    uh = torch.rand((bs, 4, 2))
    N_eval = 3
    y = torch.rand((bs, N_eval, 2))

    N = 16
    branch_encoder: Encoder = FlattenEncoder(1)
    branch: nn.Module = MLP([2*4, 32, 32, 2*N])
    trunk_encoder : Encoder = IdentityEncoder()
    trunk: nn.Module = MLP([2, 32, 32, 2*N])

    assert branch_encoder(uh).shape == (bs, 4*2)
    assert trunk_encoder(y).shape == (bs, N_eval, 2)
    assert branch(branch_encoder(uh)).shape == (bs, 2*N)
    assert trunk(trunk_encoder(y)).shape == (bs, N_eval, 2*N)

    final_bias: nn.Module | None = None
    combine_style: Literal[1,2,3,4] = 2
    sensors: torch.Tensor | None = None

    deeponet = DeepONet(branch_encoder, branch, trunk_encoder, trunk, 2, 2, final_bias, combine_style, sensors)

    vh = deeponet(uh, y)
    assert vh.shape == (bs, N_eval, 2)

    return

def test_combine3():

    """
        Omega = [0,1]x[0,1]
        v: (v1, v2) two-dimensional vector field over Omega
        u: (u1, u2) two-dimensional vector field over Omega
    """

    bs = 6
    uh = torch.rand((bs, 4, 2))
    N_eval = 3
    y = torch.rand((bs, N_eval, 2))

    N = 16
    branch_encoder: Encoder = FlattenEncoder(1)
    branch: nn.Module = MLP([2*4, 32, 32, 2*N])
    trunk_encoder : Encoder = IdentityEncoder()
    trunk: nn.Module = MLP([2, 32, 32, N])

    assert branch_encoder(uh).shape == (bs, 4*2)
    assert trunk_encoder(y).shape == (bs, N_eval, 2)
    assert branch(branch_encoder(uh)).shape == (bs, 2*N)
    assert trunk(trunk_encoder(y)).shape == (bs, N_eval, N)

    final_bias: nn.Module | None = None
    combine_style: Literal[1,2,3,4] = 3
    sensors: torch.Tensor | None = None

    deeponet = DeepONet(branch_encoder, branch, trunk_encoder, trunk, 2, 2, final_bias, combine_style, sensors)

    vh = deeponet(uh, y)
    assert vh.shape == (bs, N_eval, 2)

    return

def test_combine4():

    """
        Omega = [0,1]x[0,1]
        v: (v1, v2) two-dimensional vector field over Omega
        u: (u1, u2) two-dimensional vector field over Omega
    """

    bs = 6
    uh = torch.rand((bs, 4, 2))
    N_eval = 3
    y = torch.rand((bs, N_eval, 2))

    N = 16
    branch_encoder: Encoder = FlattenEncoder(1)
    branch: nn.Module = MLP([2*4, 32, 32, N])
    trunk_encoder : Encoder = IdentityEncoder()
    trunk: nn.Module = MLP([2, 32, 32, 2*N])

    assert branch_encoder(uh).shape == (bs, 4*2)
    assert trunk_encoder(y).shape == (bs, N_eval, 2)
    assert branch(branch_encoder(uh)).shape == (bs, N)
    assert trunk(trunk_encoder(y)).shape == (bs, N_eval, 2*N)

    final_bias: nn.Module | None = None
    combine_style: Literal[1,2,3,4] = 4
    sensors: torch.Tensor | None = None

    deeponet = DeepONet(branch_encoder, branch, trunk_encoder, trunk, 2, 2, final_bias, combine_style, sensors)

    vh = deeponet(uh, y)
    assert vh.shape == (bs, N_eval, 2)

    return

if __name__ == "__main__":
    test_combine2()
    test_combine3()
    test_combine4()
