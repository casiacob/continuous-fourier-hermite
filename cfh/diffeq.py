import jax.numpy as jnp
from typing import Callable, Tuple
from jax import lax


def rk4(ode: Callable, state: jnp.ndarray, param: Tuple, step: float) -> jnp.ndarray:
    """
    4-th order Runge Kutta integration step
    :param ode: ordinary differential equation function
    :param state: current state of the ode
    :param param: parameters to pass to ode
    :param step: integration step size
    :return: next state of the ode
    """
    k1 = ode(state, param) * step
    k2 = ode(state + 0.5 * k1, param) * step
    k3 = ode(state + 0.5 * k2, param) * step
    k4 = ode(state + step * k3, param) * step
    return state + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_ode(
    ode: Callable,
    initial_state: jnp.ndarray,
    integration_step: float,
    params: Tuple = None,
    steps_number: int = None,
    reverse: bool = False,
) -> jnp.ndarray:
    """
    Integrate ordinary differential equation over time
    The integration time is defined either by the shape of the parameters or the steps number
    :param ode: ordinary differential equation function
    :param initial_state: initial condition of the ode
    :param integration_step: integration step size
    :param params: array of parameters to pass to ode
    :param steps_number: number of integration steps
    :param reverse: if set True the integration is reversed; not that in this case the integration step should be negative
    :return: returns an array of states resulting from the integration
    """

    def integrate_body(carry, inp):
        carry = rk4(ode, carry, inp, integration_step)
        return carry, carry

    _, states = lax.scan(
        integrate_body, initial_state, params, reverse=reverse, length=steps_number
    )
    return states
