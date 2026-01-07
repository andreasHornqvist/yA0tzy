import unittest


class TestModelKinds(unittest.TestCase):
    def test_both_model_kinds_forward(self) -> None:
        import torch

        from yatzy_az.model.net import F, YatzyNet, YatzyNetConfig

        x = torch.zeros((4, F), dtype=torch.float32)

        m_res = YatzyNet(YatzyNetConfig(hidden=32, blocks=2, kind="residual"))
        logits, v = m_res(x)
        self.assertEqual(tuple(logits.shape), (4, 47))
        self.assertEqual(tuple(v.shape), (4,))

        m_mlp = YatzyNet(YatzyNetConfig(hidden=32, blocks=2, kind="mlp"))
        logits, v = m_mlp(x)
        self.assertEqual(tuple(logits.shape), (4, 47))
        self.assertEqual(tuple(v.shape), (4,))


if __name__ == "__main__":
    unittest.main()


