"""Compare the inspectable hybrid controller on a scalar linear plant."""
from hybrid_controller import ControllerConfig, simulate

config = ControllerConfig(prior_gain=0.75, residual_gain=0.04, residual_limit=0.15, action_limit=1.0)
result = simulate(initial_state=1.0, steps=20, plant_a=1.0, plant_b=1.0, config=config)
print("Hybrid LQR-prior + bounded residual; synthetic plant only")
print("initial/final state:", result["states"][0], result["states"][-1])
print("cumulative stage cost:", round(sum(result["stage_costs"]), 6))
print("first five controls:", result["actions"][:5])
