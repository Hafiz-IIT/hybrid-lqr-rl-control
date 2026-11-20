"""Inspectable hybrid controller: scalar linear feedback plus a fixed bounded cubic residual.

A scalar linear plant is used to make the control loop reproducible. No
hardware, learned policy, or formal stability guarantee is claimed.
"""
from dataclasses import dataclass
import math


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite number")
    return normalized


@dataclass(frozen=True)
class ControllerConfig:
    prior_gain: float = 0.9
    residual_gain: float = 0.05
    residual_limit: float = 0.2
    action_limit: float = 1.0

    def __post_init__(self):
        values = (self.prior_gain, self.residual_gain, self.residual_limit, self.action_limit)
        for name, value in zip(("prior_gain", "residual_gain", "residual_limit", "action_limit"), values):
            _finite(name, value)
        if self.residual_limit < 0 or self.action_limit <= 0:
            raise ValueError("residual_limit must be non-negative and action_limit positive")


def linear_feedback_prior(state: float, gain: float = 0.9) -> float:
    state = _finite("state", state)
    gain = _finite("gain", gain)
    return -gain * state


def fixed_cubic_residual(state: float, coefficient: float = 0.05, limit: float = 0.2) -> float:
    state = _finite("state", state)
    coefficient = _finite("coefficient", coefficient)
    limit = _finite("limit", limit)
    if limit < 0:
        raise ValueError("limit must be non-negative")
    raw = coefficient * state * state * state
    return max(-limit, min(limit, raw))


def action(state: float, config: ControllerConfig | None = None) -> float:
    config = ControllerConfig() if config is None else config
    if not isinstance(config, ControllerConfig):
        raise TypeError("config must be a ControllerConfig")
    raw = linear_feedback_prior(state, config.prior_gain) + fixed_cubic_residual(
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
    initial_state = _finite("initial_state", initial_state)
    plant_a = _finite("plant_a", plant_a)
    plant_b = _finite("plant_b", plant_b)
    disturbance = _finite("disturbance", disturbance)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    config = ControllerConfig() if config is None else config
    if not isinstance(config, ControllerConfig):
        raise TypeError("config must be a ControllerConfig")
    states, controls, costs = [initial_state], [], []
    for _ in range(steps):
        x = states[-1]
        u = action(x, config)
        controls.append(u)
        cost = x * x + 0.01 * u * u
        next_state = plant_a * x + plant_b * u + disturbance
        if not math.isfinite(cost) or not math.isfinite(next_state):
            raise OverflowError("simulation produced a non-finite state or cost")
        costs.append(cost)
        states.append(next_state)
    return {"states": states, "actions": controls, "stage_costs": costs}
