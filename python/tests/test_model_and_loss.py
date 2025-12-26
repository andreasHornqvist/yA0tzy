import pytest

torch = pytest.importorskip("torch")


def test_forward_shapes():
    from yatzy_az.model import A, F, YatzyNet, YatzyNetConfig

    m = YatzyNet(YatzyNetConfig(hidden=64, blocks=2))
    x = torch.randn(8, F)
    logits, v = m(x)
    assert tuple(logits.shape) == (8, A)
    assert tuple(v.shape) == (8,)
    assert torch.all(v >= -1.0001)
    assert torch.all(v <= 1.0001)


def test_loss_is_finite_and_backward_works():
    import torch.nn.functional as Fnn

    from yatzy_az.model import A, F, YatzyNet, YatzyNetConfig

    m = YatzyNet(YatzyNetConfig(hidden=64, blocks=1))
    x = torch.randn(4, F)
    logits, v = m(x)
    pi = torch.softmax(torch.randn(4, A), dim=1)
    z = torch.tanh(torch.randn(4))

    logp = Fnn.log_softmax(logits, dim=1)
    loss_pi = -(pi * logp).sum(dim=1).mean()
    loss_v = Fnn.mse_loss(v, z)
    loss = loss_pi + loss_v

    assert torch.isfinite(loss).item()
    loss.backward()


def test_one_step_changes_weights():
    import torch.nn.functional as Fnn

    from yatzy_az.model import A, F, YatzyNet, YatzyNetConfig

    m = YatzyNet(YatzyNetConfig(hidden=64, blocks=1))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)

    x = torch.randn(8, F)
    pi = torch.softmax(torch.randn(8, A), dim=1)
    z = torch.tanh(torch.randn(8))

    before = [p.detach().clone() for p in m.parameters()]
    logits, v = m(x)
    logp = Fnn.log_softmax(logits, dim=1)
    loss = (-(pi * logp).sum(dim=1).mean()) + Fnn.mse_loss(v, z)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    after = list(m.parameters())
    assert any(
        not torch.allclose(b, a.detach(), atol=0, rtol=0)
        for b, a in zip(before, after, strict=True)
    )


def test_checkpoint_roundtrip(tmp_path):
    from yatzy_az.model import F, YatzyNet, YatzyNetConfig

    m = YatzyNet(YatzyNetConfig(hidden=64, blocks=2))
    x = torch.randn(2, F)
    y0 = m(x)

    p = tmp_path / "ckpt.pt"
    torch.save({"model": m.state_dict(), "config": {"hidden": 64, "blocks": 2}}, p)

    m2 = YatzyNet(YatzyNetConfig(hidden=64, blocks=2))
    d = torch.load(p, map_location="cpu")
    m2.load_state_dict(d["model"])
    y1 = m2(x)

    assert torch.allclose(y0[0], y1[0])
    assert torch.allclose(y0[1], y1[1])
