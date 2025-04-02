from typing import NamedTuple, Callable
import jax.numpy as jnp


class OCP(NamedTuple):
    """
    continuous time, control affine optimal control problem
    J = h(x, tf) + int l(x, u, t) dt
    dxdt = f(x, u, t)
    """

    final_cost: Callable
    stage_cost: Callable
    dynamics: Callable
    final_cov: jnp.ndarray
    joint_cov: jnp.ndarray
    dim_x: int
    dim_u: int
    dt: float


class LQT(NamedTuple):
    F: jnp.ndarray
    L: jnp.ndarray
    c: jnp.ndarray
    H: jnp.ndarray
    X: jnp.ndarray
    U: jnp.ndarray
    r: jnp.ndarray
    Hf: jnp.ndarray
    Xf: jnp.ndarray
    rf: jnp.ndarray
    t0: float
    tf: float
    dt: float
    t_eval: jnp.ndarray
    x0: jnp.ndarray
