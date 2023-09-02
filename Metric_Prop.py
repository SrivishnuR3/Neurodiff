import numpy as np
from neurodiffeq import diff
from neurodiffeq.conditions import IVP
from neurodiffeq.ode import solve_system
import torch
import matplotlib.pyplot as plt

# Define Einstein's equations for a 4D spherically symmetric metric
def einstein_eqs(u1, u2, r):
    alpha, beta = u1, u2
    d_alpha_r = diff(alpha, r)
    d_beta_r = diff(beta, r)
    d2_alpha_r2 = diff(alpha, r, order=2)
    d2_beta_r2 = diff(beta, r, order=2)
    
    R_tt = torch.exp(2*beta) * (2*r*d_beta_r + r**2 * d_beta_r**2 - r**2 * d_alpha_r * d_beta_r + r**2 * d2_alpha_r2)
    R_rr = -torch.exp(2*alpha) * (2*r*d_alpha_r + r**2 * d_alpha_r**2 - r**2 * d_alpha_r * d_beta_r - r**2 * d2_beta_r2)
    
    return [R_tt, R_rr]

# Initial conditions
initial_conditions = [
    IVP(t_0=0.01, u_0=torch.tensor(0.0)),  # Start slightly away from 0 to avoid singularity
    IVP(t_0=0.01, u_0=torch.tensor(0.0))   
]

# Solve the ODE system
solution, _ = solve_system(
    ode_system=einstein_eqs, 
    conditions=initial_conditions, 
    t_min=0.01,  # Start slightly away from 0 to avoid singularity
    t_max=2.0,   # You can adjust this
    max_epochs=5000
)

# Plotting the solutions
r_vals = np.linspace(0.01, 2.0, 100)
alpha_sol, beta_sol = solution(r_vals, to_numpy=True)

plt.figure()
plt.plot(r_vals, alpha_sol, label='Alpha solution')
plt.plot(r_vals, beta_sol, label='Beta solution')
plt.ylabel('Function Value')
plt.xlabel('r')
plt.title('Einstein Equations Solutions')
plt.legend()
plt.show()
