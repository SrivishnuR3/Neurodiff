import numpy as np
from neurodiffeq import diff
from neurodiffeq.conditions import IVP
from neurodiffeq.ode import solve_system, Monitor
import torch
import numpy as np
import matplotlib.pyplot as plt

def einstein_eqs(alpha, beta, r):
    d_alpha_r = diff(alpha, r)
    d_beta_r = diff(beta, r)
    d2_alpha_r2 = diff(alpha, r, order=2)
    d2_beta_r2 = diff(beta, r, order=2)
    
    R_tt = torch.exp(2*beta) * (2*r*d_beta_r + r**2 * d_beta_r**2 - r**2 * d_alpha_r * d_beta_r + r**2 * d2_alpha_r2)
    R_rr = -torch.exp(2*alpha) * (2*r*d_alpha_r + r**2 * d_alpha_r**2 - r**2 * d_alpha_r * d_beta_r - r**2 * d2_beta_r2)
    
    return [R_tt, R_rr]

initial_conditions = [
    IVP(t_0=0, u_0=torch.tensor(0.0)),  # initial condition for alpha at r=0
    IVP(t_0=0, u_0=torch.tensor(0.0))   # initial condition for beta at r=0
]

solution = solve_system(
    ode_system=einstein_eqs, 
    conditions=initial_conditions, 
    t_min=0.0, 
    t_max=2.0  # You can adjust this
)

alpha_sol, beta_sol = solution

ts = np.linspace(0, 2.0, 50)
plt.figure()
plt.plot(ts, alpha_sol, label='Alpha')
plt.plot(ts, beta_sol, '.', label='Beta')
plt.ylabel('u')
plt.xlabel('r')
plt.title('comparing solutions')
plt.legend()
plt.show()