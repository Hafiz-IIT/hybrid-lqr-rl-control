"""Inspectable hybrid controller: LQR-like prior plus bounded residual.

A scalar linear plant is used to make the control loop reproducible. No
hardware, learned policy, or formal stability guarantee is claimed.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ControllerConfig:
    prior_gain: float = 0.9
    residual_gain: float = 0.05
    residual_limit: float = 0.2
    action_limit: float = 1.0

    def __post_init__(self):
        values = (self.prior_gain, self.residual_gain, self.residual_limit, self.action_limit)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("controller parameters must be finite")
        if self.residual_limit < 0 or self.action_limit <= 0:
            raise ValueError("residual_limit must be non-negative and action_limit positive")


def lqr_prior(state: float, gain: float = 0.9) -> float:
    if not math.isfinite(state) or not math.isfinite(gain):
        raise ValueError("state and gain must be finite")
    return -gain * state


def learned_residual(state: float, coefficient: float = 0.05, limit: float = 0.2) -> float:
    if not all(math.isfinite(v) for v in (state, coefficient, limit)) or limit < 0:
        raise ValueError("finite inputs and non-negative limit required")
    raw = coefficient * state ** 3
    return max(-limit, min(limit, raw))


def action(state: float, config: ControllerConfig | None = None) -> float:
    config = config or ControllerConfig()
    raw = lqr_prior(state, config.prior_gain) + learned_residual(
        state, config.residual_gain, config.residual_limit
    )
    return max(-config.action_limit, min(config.action_limit, raw))


def simulate(
    initial_state: float,
    steps: int = 50,
    plant_a: float = 1.0,
    plant_b: float = 1.0,
    config: ControllerConfig | None = None,
    disturbance: float = 0.0,
) -> dict[str, list[float]]:
    """Simulate x[t+1] = plant_a*x[t] + plant_b*u[t] + disturbance."""
    if steps < 0 or not all(math.isfinite(v) for v in (initial_state, plant_a, plant_b, disturbance)):
        raise ValueError("steps must be non-negative and plant parameters finite")
    config = config or ControllerConfig()
    states, controls, costs = [initial_state], [], []
    for _ in range(steps):
        x = states[-1]
        u = action(x, config)
        controls.append(u)
        costs.append(x * x + 0.01 * u * u)
        states.append(plant_a * x + plant_b * u + disturbance)
    return {"states": states, "actions": controls, "stage_costs": costs}
