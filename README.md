# Hybrid LQR + Learned Residual Control

> **Research prototype:** combining an interpretable LQR-style baseline with a residual correction layer.

## Research question
Can a structured linear controller provide a useful prior while a learned residual handles dynamics not captured by the prior?

## Implemented
- scalar linear feedback baseline
- residual correction interface
- compositional action calculation
- deterministic rollout
- tests

## Quickstart
```bash
python demo.py
python -m unittest discover -s tests -v
```

## Boundary
The current artifact is a research scaffold. It does not claim optimal control, learned-policy superiority, formal safety, or robotic deployment.

Related: [Safe RL Action Gate](https://github.com/Hafiz-IIT/safe-rl-action-gate) · [Residual RL with Linear Priors](https://github.com/Hafiz-IIT/residual-rl-linear-priors)
\n\n## Evidence status\n\nThis repository is the **primary executable control scaffold**; its manuscript companion is [hybrid-lqr-rl-paper](https://github.com/Hafiz-IIT/hybrid-lqr-rl-paper).\n\n## Limitations\n\nThe current implementation is intentionally scalar and deterministic. It demonstrates composition of a transparent LQR-style prior and residual interface, but does not establish optimality, broad benchmark superiority, formal safety, or robotic deployment.\n