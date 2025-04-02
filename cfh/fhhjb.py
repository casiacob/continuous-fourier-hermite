import jax.numpy as jnp
import jax.scipy as jsp
from jax import debug, hessian, grad, jacfwd
from jax import vmap
from jax import lax
from cfh.diffeq import integrate_ode
from cfh.objects import OCP
from cfh.gh import expected_hessian, expected_gradient


def backward_pass(nominal_states: jnp.ndarray, nominal_controls: jnp.ndarray, ocp: OCP):
    final_state = nominal_states[-1]
    S_final = expected_hessian(final_state, ocp.final_cov, ocp.final_cost)
    v_final = expected_gradient(final_state, ocp.final_cov, ocp.final_cost)
    S_final_flat = jnp.ravel(S_final)

    def hamiltonian(xu, params):
        x = xu[: ocp.dim_x]
        u = xu[ocp.dim_x :]
        S, v, x_hat = params
        dx = x - x_hat
        return ocp.stage_cost(x, u) + (v + S @ dx).T @ ocp.dynamics(x, u)

    def dVdt(V, params):
        v = V[: ocp.dim_x]
        S = V[ocp.dim_x :]
        S = S.reshape(ocp.dim_x, ocp.dim_x)
        x_hat, u_hat = params
        xu = jnp.hstack((x_hat, u_hat))

        # evaluate hamiltonian expected gradient and Hessian
        H_hessian = expected_hessian(
            xu, ocp.joint_cov, hamiltonian, params=(S, v, x_hat)
        )
        H_gradient = expected_gradient(
            xu, ocp.joint_cov, hamiltonian, params=(S, v, x_hat)
        )

        # split result
        Hxx = H_hessian[: ocp.dim_x, : ocp.dim_x]
        Hxu = H_hessian[: ocp.dim_x, ocp.dim_x :]
        Huu = H_hessian[ocp.dim_x :, ocp.dim_x :]
        Hux = H_hessian[ocp.dim_x :, : ocp.dim_x]
        Hx = H_gradient[: ocp.dim_x]
        Hu = H_gradient[ocp.dim_x :]

        # feed forward and feedback gain
        d = -jsp.linalg.solve(Huu, Hu)
        K = -jsp.linalg.solve(Huu, Hux)

        # differential equations
        dvdt = -(Hx + K.T @ Hu + K.T @ Huu @ d + Hxu @ d)
        dSdt = -(Hxx + K.T @ Huu @ K + K.T @ Hux + Hxu @ K)
        dSdt = jnp.ravel(dSdt)
        return jnp.hstack((dvdt, dSdt))

    val_fun_params = integrate_ode(
        dVdt,
        jnp.hstack((v_final, S_final_flat)),
        -ocp.dt,
        (nominal_states[:-1], nominal_controls),
        reverse=True,
    )
    v_array = val_fun_params[:, : ocp.dim_x]
    S_array = val_fun_params[:, ocp.dim_x :]
    S_array = S_array.reshape(-1, ocp.dim_x, ocp.dim_x)

    def compute_gains(v, S, x_hat, u_hat):
        xu = jnp.hstack((x_hat, u_hat))

        # evaluate hamiltonian expected gradient and Hessian
        H_hessian = expected_hessian(
            xu, ocp.joint_cov, hamiltonian, params=(S, v, x_hat)
        )
        H_gradient = expected_gradient(
            xu, ocp.joint_cov, hamiltonian, params=(S, v, x_hat)
        )

        # split result
        Hxu = H_hessian[: ocp.dim_x, ocp.dim_x :]
        Huu = H_hessian[ocp.dim_x :, ocp.dim_x :]
        Hu = H_gradient[ocp.dim_x :]

        # feed forward and feedback gain
        d = -jsp.linalg.solve(Huu, Hu)
        K = -jsp.linalg.solve(Huu, Hxu.T)

        return d, K

    d_array, K_array = vmap(compute_gains)(
        v_array, S_array, nominal_states[:-1], nominal_controls
    )
    return v_array, S_array, d_array, K_array


def forward_pass(
    ff_gain: jnp.ndarray,
    gain: jnp.ndarray,
    nominal_states: jnp.ndarray,
    nominal_controls: jnp.ndarray,
    ocp: OCP,
):
    def closed_loop_dynamics(x, params):
        d, K, x_hat, u_hat = params
        dx = x - x_hat
        du = d + K @ dx
        return ocp.dynamics(x, u_hat + du)

    new_nominal_states = integrate_ode(
        closed_loop_dynamics,
        nominal_states[0],
        ocp.dt,
        (ff_gain, gain, nominal_states[:-1], nominal_controls),
    )
    new_nominal_states = jnp.vstack((nominal_states[0], new_nominal_states))

    def control_updates(d, K, x_new, x_old, u_old):
        dx = x_new - x_old
        return u_old + d + K @ dx

    new_nominal_controls = vmap(control_updates)(
        ff_gain, gain, new_nominal_states[:-1], nominal_states[:-1], nominal_controls
    )
    return new_nominal_states, new_nominal_controls


def fhddp(nominal_states: jnp.ndarray, nominal_controls: jnp.ndarray, ocp: OCP):

    def while_body(val):
        xn, un, iteration = val
        v_fh, S_fh, d_fh, K_fh = backward_pass(xn, un, ocp)
        x_fh, u_fh = forward_pass(d_fh, K_fh, xn, un, ocp)
        iteration += 1
        return x_fh, u_fh, iteration

    def while_cond(val):
        _, _, iterations = val
        exit_condition = iterations > 1
        return jnp.logical_not(exit_condition)

    return lax.while_loop(while_cond, while_body, (nominal_states, nominal_controls, 0))
