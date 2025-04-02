import jax.numpy as jnp
import jax.scipy as jsp
from cfh.objects import LQT
from cfh.diffeq import integrate_ode
from jax import vmap


def bwd_pass(lqt: LQT) -> (jnp.ndarray, jnp.ndarray):
    S_final = lqt.Hf.T @ lqt.Xf @ lqt.Hf
    v_final = lqt.Hf.T @ lqt.Xf @ lqt.rf

    def dSdt(S, param):
        F, L, H, X, U = param
        return (
            -F.T @ S
            - S @ F
            - H.T @ X @ H
            + S @ L @ jsp.linalg.solve(U, L.T @ S, assume_a="pos")
        )

    def dvdt(v, param):
        F, L, c, H, X, U, r, S = param
        return (
            -H.T @ X @ r
            + S @ c
            - F.T @ v
            + S @ L @ jsp.linalg.solve(U, L.T @ v, assume_a="pos")
        )

    S_array = integrate_ode(
        dSdt, S_final, -lqt.dt, params=(lqt.F, lqt.L, lqt.H, lqt.X, lqt.U), reverse=True
    )
    v_array = integrate_ode(
        dvdt,
        v_final,
        -lqt.dt,
        params=(
            lqt.F,
            lqt.L,
            lqt.c,
            lqt.H,
            lqt.X,
            lqt.U,
            lqt.r,
            jnp.vstack((S_array[1:], S_final[jnp.newaxis, :, :])),
        ),
        reverse=True,
    )
    return S_array, v_array


def fwd_pass(
    lqt: LQT, S_array: jnp.ndarray, v_array: jnp.ndarray
) -> (jnp.ndarray, jnp.ndarray):
    F_cl = vmap(
        lambda F, L, U, S: F - L @ jsp.linalg.solve(U, L.T @ S, assume_a="pos")
    )(lqt.F, lqt.L, lqt.U, S_array)

    c_cl = vmap(
        lambda L, c, U, v: L @ jsp.linalg.solve(U, L.T @ v, assume_a="pos") + c
    )(lqt.L, lqt.c, lqt.U, v_array)

    def dxdt_cl(x, param):
        F, c = param
        return F @ x + c

    optimal_trajectory = integrate_ode(dxdt_cl, lqt.x0, lqt.dt, params=(F_cl, c_cl))
    optimal_trajectory = jnp.vstack((lqt.x0, optimal_trajectory))
    optimal_control = vmap(
        lambda R, B, S, v, x: -jsp.linalg.solve(R, B.T @ S @ x)
        + jsp.linalg.solve(R, B.T @ v)
    )(lqt.U, lqt.L, S_array, v_array, optimal_trajectory[:-1])
    return optimal_trajectory, optimal_control
