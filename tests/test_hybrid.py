import unittest
from hybrid_controller import lqr_prior, learned_residual, action

class HybridTests(unittest.TestCase):
    def test_composition(self):
        x=.5
        self.assertAlmostEqual(action(x),lqr_prior(x)+learned_residual(x))

if __name__=="__main__": unittest.main()
