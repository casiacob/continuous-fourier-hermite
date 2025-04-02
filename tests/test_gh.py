from cfh.gh import (
    evaluate_expectation_1d,
    evaluate_expectation_nd,
    expected_gradient,
    expected_hessian,
)
from jax import grad, hessian
import unittest
import jax.numpy as jnp
import math
from jax import config
import jax.random as random

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")


def gaussian_moments(m: int):
    m_over_2 = int(m / 2)
    return jnp.where(
        jnp.bool_(m % 2 == 0),
        2 ** (-m_over_2) * math.factorial(m) / math.factorial(m_over_2),
        0,
    )


class Test_one_d_gauss_hermite(unittest.TestCase):
    def test_gh_1d_gaussian_moment2(self):
        I2 = evaluate_expectation_1d(0.0, 1.0, lambda x: x**2, 4)
        gm2 = gaussian_moments(2)
        err2 = jnp.abs(I2 - gm2)
        self.assertTrue(err2 < 1e-10)

    def test_gh_1d_gaussian_moment3(self):
        I3 = evaluate_expectation_1d(0.0, 1.0, lambda x: x**3, 4)
        gm3 = gaussian_moments(3)
        err3 = jnp.abs(I3 - gm3)
        self.assertTrue(err3 < 1e-10)

    def test_gh_1d_gaussian_moment4(self):
        I4 = evaluate_expectation_1d(0.0, 1.0, lambda x: x**4, 4)
        gm4 = gaussian_moments(4)
        err4 = jnp.abs(I4 - gm4)
        self.assertTrue(err4 < 1e-10)

    def test_gh_1d_gaussian_moment5(self):
        I5 = evaluate_expectation_1d(0.0, 1.0, lambda x: x**5, 4)
        gm5 = gaussian_moments(5)
        err5 = jnp.abs(I5 - gm5)
        self.assertTrue(err5 < 1e-10)


class Test_n_d_gauss_hermite(unittest.TestCase):
    def generate_test_params(self):
        n = 10

        key = random.PRNGKey(1)

        key1, key2, key3 = random.split(key, 3)

        # Generate a random symmetric matrix S
        A = random.normal(key1, shape=(n, n))
        S = (A + A.T) / 2  # Make it symmetric

        # Generate a random mean vector m
        m = random.normal(key2, shape=(n,))

        # Generate a random positive definite covariance matrix P
        B = random.normal(key3, shape=(n, n))
        P = B @ B.T + n * jnp.eye(n)  # Ensure positive definiteness

        return S, m, P

    def test_multivariate_gaussian(self):
        S, m, P = self.generate_test_params()
        I = evaluate_expectation_nd(m, P, lambda x, p: x.T @ S @ x)
        expectation = jnp.trace(S @ P) + m.T @ S @ m
        err = jnp.abs(I - expectation)
        self.assertTrue(err < 1e-10)


class Test_gradient_and_Hessian_expectation(unittest.TestCase):
    def quadratic_function(self):
        n = 10
        key = random.PRNGKey(134)
        key1, key2, key3 = random.split(key, 3)

        # Generate a random symmetric matrix S
        A = random.normal(key1, shape=(n, n))
        S = (A + A.T) / 2  # Make it symmetric

        # Generate a random mean vector v
        v = random.normal(key2, shape=(n,))

        # Generate an evaluation point
        x_eval = random.normal(key3, shape=(n,))
        return S, v, x_eval, 1e-3 * jnp.eye(n)

    def test_gradient(self):
        S, v, x_eval, cov = self.quadratic_function()
        grad_jax = grad(lambda x: 0.5 * x.T @ S @ x + x.T @ v)(x_eval)
        grad_fh = expected_gradient(
            x_eval, cov, lambda x, p: 0.5 * x.T @ S @ x + x.T @ v
        )
        err = jnp.max(jnp.abs(grad_jax - grad_fh))
        self.assertTrue(err < 1e-10)

    def test_Hessian(self):
        S, v, x_eval, cov = self.quadratic_function()
        hess_jax = hessian(lambda x: 0.5 * x.T @ S @ x + x.T @ v)(x_eval)
        hess_fh = expected_hessian(
            x_eval, cov, lambda x, p: 0.5 * x.T @ S @ x + x.T @ v
        )
        err = jnp.max(jnp.abs(hess_jax - hess_fh))
        self.assertTrue(err < 1e-10)
