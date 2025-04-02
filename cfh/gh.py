import jax.numpy as jnp
import jax.scipy as jsp
from jax import vmap
from jax import lax
from jax import config
import math
from typing import Callable, Tuple
from jax import debug

config.update("jax_enable_x64", True)


def herm_roots(p: int) -> jnp.ndarray:
    """
    Compute the roots of the Hermite polynomial of order p, H_{p}(x), as eigenvalues of tridiagonal matrix
    :param p: Order of hermite polynomial
    :return: Roots of Hermite polynomial H_{p}(x)
    """
    d = jnp.zeros(p)
    e = jnp.sqrt(jnp.arange(1, p))
    return jsp.linalg.eigh_tridiagonal(d, e, eigvals_only=True)


def herm_coeff(p: int) -> jnp.ndarray:
    """
    Compute the coefficients of the Hermite polynomial of order p-1, H_{p-1}(x)
    :param p: Order of hermite polynomial H_{p-1}(x)
    :return: Coefficients of Hermite polynomial H_{p-1}(x)
    """
    z = jnp.zeros(p)
    H0 = z.at[-1].set(1.0)
    H1 = z.at[-2].set(1.0)

    def body(carry, inp):
        H_n_minus_1, H_n_minus_2, it_cnt = carry
        Hn = jnp.hstack((H_n_minus_1[1:], jnp.array([0.0]))) - it_cnt * H_n_minus_2
        it_cnt += 1
        return (Hn, H_n_minus_1, it_cnt), Hn

    _, H = lax.scan(body, (H1, H0, 1), xs=None, length=p - 2)

    return H[-1]


def weights_1d(xi: jnp.ndarray, H: jnp.ndarray, p: int):
    """
    Generate unit weights
    :param xi: Sigma points as roots of Hermite polynomial
    :param H: Coefficients of Hermite polynomial
    :param p: Order of Hermite polynomial
    :return: Integration weights
    """
    H_at_xi = jnp.polyval(H, xi)
    return math.factorial(p) / p**2 / H_at_xi**2


def weights_nd(xi: jnp.ndarray, H: jnp.ndarray, n: int, p: int):
    """
    Form multidimensional weights as products of one-dimensional weights
    :param xi: Sigma points as roots of Hermite polynomial
    :param H: Coefficients of Hermite polynomial
    :param n: Dimension
    :param p: Order of Hermite polynomial
    :return: n-dimensional weights
    """
    w_1d = weights_1d(xi, H, p)
    grids = jnp.meshgrid(*([w_1d] * n))
    combinations = jnp.stack(grids, axis=-1).reshape(-1, n)
    w_nd = jnp.prod(combinations, axis=1)
    return w_nd


def sigma_points_nd(xi: jnp.ndarray, n: int):
    """
    Form multidimensional sigma points as Cartesian product of the onedimensional unit sigma points
    :param xi: One-dimensional sigma points
    :param n: Dimension
    :return: Array of n-dimensional sigma points
    """
    grids = jnp.meshgrid(*[xi] * n)  # Create n-dimensional grid
    xi_nd = jnp.stack(grids, axis=-1).reshape(-1, n)
    return xi_nd


def evaluate_expectation_1d(m: float, P: float, g: Callable, p: int):
    """
    p-th order Gauss-Hermite approximation to the one dimensional integral
    int g(x) N(x | m, P) dx
    :param m: Gaussain mean
    :param P: Gaussian variance
    :param g: scalar function g : R -> R
    :param p: Order of approximation
    :return: expected value
    """
    xi = herm_roots(p)
    H = herm_coeff(p)
    w = weights_1d(xi, H, p)
    g_eval = vmap(lambda x: g(m + jnp.sqrt(P) * x))(xi)
    return jnp.sum(w * g_eval)


def evaluate_expectation_nd(
    m: jnp.ndarray, P: jnp.ndarray, g: Callable, params: Tuple = None, p: int = 3
):
    """
    p-th order Gauss-Hermite approximation to the n-dimensional integral
    int g(x) N(x | m, P) dx
    :param m: Gaussian mean of dimension n by 1
    :param P: Gaussian covariance of dimension n by n
    :param g: scalar function g : R^n -> R
    :param params: Tuple of parameters to pass to g
    :param p: order of approximation
    :return: expected value
    """
    n = m.shape[0]
    chol = jnp.linalg.cholesky(P)
    xi = herm_roots(p)
    H = herm_coeff(p)
    w_nd = weights_nd(xi, H, n, p)
    xi_nd = sigma_points_nd(xi, n)
    g_at_xi = vmap(lambda x: g(m + chol @ x, params))(xi_nd)
    return jnp.sum(vmap(lambda x, y: x * y)(w_nd, g_at_xi), axis=0)


def expected_gradient(
    m: jnp.ndarray, P: jnp.ndarray, g: Callable, params: Tuple = None, p: int = 3
):
    """
    p-th order Gauss Hermite approximation to the expected gradient of g
    :param m: Gaussian mean of dimension n by 1
    :param P: Gaussian covariance of dimension n by n
    :param g: scalar function g: R^n -> R
    :param params: Tuple of parameters to pass to g
    :param p: order of the approximation
    :return: expected gradient of dimension n by 1
    """
    n = m.shape[0]
    chol = jnp.linalg.cholesky(P)
    xi = herm_roots(p)
    H = herm_coeff(p)
    w_nd = weights_nd(xi, H, n, p)
    xi_nd = sigma_points_nd(xi, n)
    g_at_xi = vmap(lambda x: g(m + chol @ x, params) * x)(xi_nd)
    b = jnp.sum(vmap(lambda x, y: x * y)(w_nd, g_at_xi), axis=0)
    return jsp.linalg.solve(chol.T, b)


def expected_hessian(
    m: jnp.ndarray, P: jnp.ndarray, g: Callable, params: Tuple = None, p: int = 3
):
    """
    p-th order Gauss Hermite approximation to the expected Hessian of g
    :param m: Gaussian mean of dimension n by 1
    :param P: Gaussian covariance of dimension n by n
    :param g: scalar function g: R^n -> R
    :param params: Tuple of parameters to pass to g
    :param p: order of the approximation
    :return: expected Hessian of dimension n by n
    """
    n = m.shape[0]
    chol = jnp.linalg.cholesky(P)
    xi = herm_roots(p)
    H = herm_coeff(p)
    w_nd = weights_nd(xi, H, n, p)
    xi_nd = sigma_points_nd(xi, n)
    g_at_xi = vmap(lambda x: g(m + chol @ x, params) * (jnp.outer(x, x) - jnp.eye(n)))(
        xi_nd
    )
    C = jnp.sum(vmap(lambda x, y: x * y)(w_nd, g_at_xi), axis=0)
    chol_inv = jsp.linalg.solve(chol, jnp.eye(n))
    return chol_inv.T @ C @ chol_inv
