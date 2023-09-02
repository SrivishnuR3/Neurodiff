from neurodiffeq import diff
from neurodiffeq.ode import solve
from neurodiffeq.conditions import IVP
import torch
import numpy as np
import matplotlib.pyplot as plt

def simple_ode(y, t):
    return diff(y, t) - y

con = IVP(t_0=0.0, u_0=1.0)

y_sol, loss_history = solve(ode=simple_ode, condition=con, t_min=0.0, t_max=2.0)

ts = np.linspace(0, 2.0, 50)
u_net = y_sol(ts, to_numpy=True)
u_ana = np.exp(+ts)

plt.figure()
plt.plot(ts, u_net, label='ANN-based solution')
plt.plot(ts, u_ana, '.', label='analytical solution')
plt.ylabel('u')
plt.xlabel('t')
plt.title('comparing solutions')
plt.legend()
plt.show()

plt.figure()
plt.plot(loss_history['train_loss'], label='training loss')
plt.plot(loss_history['valid_loss'], label='validation loss')
plt.yscale('log')
plt.title('loss during training')
plt.legend()
plt.show()