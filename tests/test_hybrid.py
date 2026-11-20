import unittest

from hybrid_controller import (
    action,
    fixed_cubic_residual,
    linear_feedback_prior,
)


class HybridTests(unittest.TestCase):
    def test_extremely_large_integer_state_is_rejected_cleanly(self):
        with self.assertRaises(ValueError):
            action(10 ** 10000)

    def test_direct_action_rejects_invalid_config_type(self):
        with self.assertRaises(TypeError):
            action(0.5, config={})

    def test_composition_uses_fixed_residual_and_linear_feedback(self):
        state = 0.5
        self.assertAlmostEqual(
            action(state),
            linear_feedback_prior(state) + fixed_cubic_residual(state),
        )


if __name__ == "__main__":
    unittest.main()
