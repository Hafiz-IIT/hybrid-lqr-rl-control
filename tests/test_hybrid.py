import unittest

from hybrid_controller import (
    action,
    fixed_cubic_residual,
    linear_feedback_prior,
)


class HybridTests(unittest.TestCase):
    def test_composition_uses_fixed_residual_and_linear_feedback(self):
        state = 0.5
        self.assertAlmostEqual(
            action(state),
            linear_feedback_prior(state) + fixed_cubic_residual(state),
        )


if __name__ == "__main__":
    unittest.main()
