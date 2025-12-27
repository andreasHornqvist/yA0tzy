import pytest

torch = pytest.importorskip("torch")


def _save_best_checkpoint(path, model, opt_state=None):
    payload = {"model": model.state_dict()}
    if opt_state is not None:
        payload["optimizer"] = opt_state
    torch.save(payload, path)


def test_optimizer_state_is_reset_on_best_to_candidate_init(tmp_path):
    from yatzy_az.model import YatzyNet, YatzyNetConfig
    from yatzy_az.trainer.train import init_from_best

    device = torch.device("cpu")

    # Create a best model and run a step to populate optimizer moments.
    m = YatzyNet(YatzyNetConfig(hidden=32, blocks=1))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.0)
    x = torch.randn(4, 45)
    logits, v = m(x)
    loss = (logits.square().mean()) + (v.square().mean())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    assert len(opt.state) > 0

    best_path = tmp_path / "best.pt"
    # Even if best checkpoint includes an optimizer state, it must be ignored.
    _save_best_checkpoint(best_path, m, opt.state_dict())

    cand_model, cand_opt, start_step = init_from_best(
        best_path=best_path, hidden=32, blocks=1, lr=1e-3, weight_decay=0.0, device=device
    )

    assert start_step == 0
    assert len(cand_opt.state) == 0  # boundary requirement

    # Sanity: weights are equal after init.
    for (n0, p0), (n1, p1) in zip(cand_model.named_parameters(), m.named_parameters(), strict=True):
        assert n0 == n1
        assert torch.allclose(p0.detach(), p1.detach())


def test_resume_loads_optimizer_state(tmp_path):
    from yatzy_az.model import YatzyNet, YatzyNetConfig
    from yatzy_az.trainer.train import resume_candidate

    device = torch.device("cpu")

    m = YatzyNet(YatzyNetConfig(hidden=32, blocks=1))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.0)
    x = torch.randn(4, 45)
    logits, v = m(x)
    loss = (logits.square().mean()) + (v.square().mean())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    assert len(opt.state) > 0

    cand_path = tmp_path / "candidate.pt"
    torch.save(
        {"model": m.state_dict(), "optimizer": opt.state_dict(), "meta": {"train_step": 7}},
        cand_path,
    )

    m2, opt2, step = resume_candidate(
        candidate_path=cand_path, hidden=32, blocks=1, lr=1e-2, weight_decay=0.0, device=device
    )
    assert step == 7
    assert len(opt2.state) > 0

    # Sanity: weights match.
    for (n0, p0), (n1, p1) in zip(m2.named_parameters(), m.named_parameters(), strict=True):
        assert n0 == n1
        assert torch.allclose(p0.detach(), p1.detach())
