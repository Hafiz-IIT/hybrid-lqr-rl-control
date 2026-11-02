# Hybrid LQR + Residual Control

An inspectable scalar control experiment combining a transparent linear feedback prior with a bounded nonlinear residual term on a simple discrete-time plant.

## Research question
Can a bounded nonlinear residual complement a transparent linear feedback prior without allowing the residual to dominate the controller?

## Implemented
- linear feedback prior u_p = -k_p x
- cubic residual with explicit clipping
- final action saturation
- configurable gains and bounds
- scalar plant rollout with constant disturbance
- state, action, and stage-cost traces

## Run
    python demo.py
    python -m unittest discover -s tests -v

## Plant and objective
The simulation uses x[t+1] = a*x[t] + b*u[t] + w and illustrative stage cost x[t]^2 + 0.01*u[t]^2. These choices are not a calibrated physical robot model.

## Limitations
The residual is a fixed nonlinear function, not a trained RL policy. The implementation does not solve a Riccati equation or provide formal stability analysis. Bounded actions do not imply closed-loop safety or stability.

See [EVALUATION.md](EVALUATION.md) for the matched-baseline experiment design.
