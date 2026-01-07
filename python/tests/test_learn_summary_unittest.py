import unittest


class TestLearnSummaryHelpers(unittest.TestCase):
    def test_parse_shard_idx(self):
        from yatzy_az.trainer import train as tr

        self.assertEqual(tr._parse_shard_idx("shard_000123.safetensors"), 123)
        self.assertEqual(tr._parse_shard_idx("shard_12.meta.json"), 12)
        self.assertIsNone(tr._parse_shard_idx("nope_000001.safetensors"))
        self.assertIsNone(tr._parse_shard_idx("shard_abc.safetensors"))

    def test_weighted_quantile_basic(self):
        from yatzy_az.trainer import train as tr

        vals = [0.0, 10.0]
        w = [1.0, 1.0]
        self.assertEqual(tr._weighted_quantile(vals, w, 0.5), 0.0)  # lower median by design
        self.assertEqual(tr._weighted_quantile(vals, w, 0.95), 10.0)

    def test_learn_summary_agg_shapes_and_ece(self):
        import torch
        from yatzy_az.trainer import train as tr

        agg = tr._LearnSummaryAgg(seed=1, bins=9)

        # Two samples over 4 actions (rest treated as 0-prob).
        pi = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        v = torch.tensor([-1.0, 1.0], dtype=torch.float32)
        z = torch.tensor([-1.0, 1.0], dtype=torch.float32)
        # Dummy model log-probs (log softmax output), normalized distributions.
        p_model = torch.tensor(
            [
                [0.97, 0.01, 0.01, 0.01],
                [0.50, 0.50, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        logp = torch.log(p_model.clamp_min(1e-12))

        agg.record_step_ms(10.0)
        agg.update_batch(pi=pi, v_pred=v, z=z, logp=logp)
        out = agg.finalize()

        # Has core keys.
        self.assertIn("pi_entropy_mean", out)
        self.assertIn("pi_top1_p50", out)
        self.assertIn("pi_eff_actions_p50", out)
        self.assertIn("pi_model_entropy_mean", out)
        self.assertIn("pi_kl_mean", out)
        self.assertIn("pi_entropy_gap_mean", out)
        self.assertIn("v_pred_mean", out)
        self.assertIn("v_pred_sat_frac", out)
        self.assertIn("calibration_bins", out)
        self.assertIn("ece", out)

        # Perfect calibration for these two points.
        self.assertAlmostEqual(float(out["ece"]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()

