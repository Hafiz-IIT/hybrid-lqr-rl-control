import unittest
from hybrid_controller import ControllerConfig, action, simulate

class RolloutTests(unittest.TestCase):
    def test_action_is_bounded(self):
        cfg = ControllerConfig(action_limit=0.4)
        self.assertLessEqual(abs(action(100.0, cfg)), 0.4)

    def test_rollout_shapes(self):
        result = simulate(0.5, steps=7)
        self.assertEqual(len(result["states"]), 8)
        self.assertEqual(len(result["actions"]), 7)
        self.assertEqual(len(result["stage_costs"]), 7)

    def test_zero_steps(self):
        self.assertEqual(simulate(1.0, steps=0)["states"], [1.0])

if __name__ == "__main__": unittest.main()
