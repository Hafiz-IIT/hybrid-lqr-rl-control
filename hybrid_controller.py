def lqr_prior(state: float, gain: float = 0.9) -> float:
    return -gain * state

def learned_residual(state: float, coefficient: float = 0.05) -> float:
    return coefficient * (state ** 3)

def action(state: float) -> float:
    return lqr_prior(state) + learned_residual(state)
