from hybrid_controller import lqr_prior, learned_residual, action

state=.5
print("Hybrid LQR + residual control")
print("prior:",lqr_prior(state))
print("residual:",learned_residual(state))
print("combined:",action(state))
