import jax.numpy as jnp
import matplotlib.pyplot as plt
from cfh.objects import LQT
from cfh.clqt import bwd_pass, fwd_pass
from cfh.fhhjb import backward_pass, forward_pass, fhddp
from cfh.diffeq import integrate_ode
from cfh.objects import OCP
from jax import random
from cfh.cddp import val_fun_comp
from jax import config
from jax import debug

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

t0 = 0.0
tf = 10.0
dt = 0.01
t_eval = jnp.arange(t0, tf + dt, dt)

F = jnp.array([[0.0, 1.0], [-1.0, -2.0]])
L = jnp.array([[0.0], [1.0]])
c = jnp.array([0.0, 0.0])
H = jnp.eye(2)
X = jnp.array([[1.0, 0.0], [0.0, 2.0]])
U = jnp.array([[1.0]])
Hf = H
Xf = jnp.array([[10.0, 0.0], [0.0, 0.0]])
rf = jnp.array([0.0, 0.0])
x0 = jnp.array([5.0, 5.0])

F = jnp.kron(jnp.ones((len(t_eval) - 1, 1, 1)), F)
L = jnp.kron(jnp.ones((len(t_eval) - 1, 1, 1)), L)
c = jnp.kron(jnp.ones((len(t_eval) - 1, 1)), c)
r = jnp.kron(jnp.ones((len(t_eval) - 1, 1)), rf)
H = jnp.kron(jnp.ones((len(t_eval) - 1, 1, 1)), H)
X = jnp.kron(jnp.ones((len(t_eval) - 1, 1, 1)), X)
U = jnp.kron(jnp.ones((len(t_eval) - 1, 1, 1)), U)

lqt_problem = LQT(F, L, c, H, X, U, r, Hf, Xf, rf, t0, tf, dt, t_eval, x0)
S, v = bwd_pass(lqt_problem)
sol_x, sol_u = fwd_pass(lqt_problem, S, v)
plt.plot(t_eval, sol_x[:, 0], color="gray", linewidth=3)
plt.plot(t_eval, sol_x[:, 1], color="gray", linewidth=3)


def f(x, u):
    A = jnp.array([[0.0, 1.0], [-1.0, -2.0]])
    B = jnp.array([[0.0], [1.0]])
    return A @ x + B @ u


def h(x, params):
    xd = rf
    P = jnp.array([[10.0, 0.0], [0.0, 0.0]])
    return 0.5 * (x - xd).T @ P @ (x - xd)


def l(x, u):
    Q = jnp.array([[1.0, 0.0], [0.0, 2.0]])
    R = jnp.array([[1.0]])
    xd = rf
    return 0.5 * (x - xd).T @ Q @ (x - xd) + 0.5 * u.T @ R @ u


key = random.PRNGKey(100)
un = 0.1 * random.normal(key, shape=(len(t_eval) - 1, 1))
# un = sol_u
xn = integrate_ode(f, x0, dt, un)
xn = jnp.vstack((x0, xn))

final_cov = 1e-6 * jnp.eye(2)
joint_cov = 1e-6 * jnp.eye(3)

ocp = OCP(h, l, f, final_cov, joint_cov, 2, 1, dt)

v_fh, S_fh, d_fh, K_fh = backward_pass(xn, un, ocp)
x_fh, u_fh = forward_pass(d_fh, K_fh, xn, un, ocp)

plt.plot(t_eval, x_fh[:, 0], linestyle="dashed")
plt.plot(t_eval, x_fh[:, 1], linestyle="dashed")
plt.show()
