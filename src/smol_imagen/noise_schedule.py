"""
Code adapted from: https://github.com/LuChengTHU/dpm-solver/blob/835411b66f7e9506820b0cc86ccec4d54cd2abd6/dpm_solver_pytorch.py
"""

from abc import abstractmethod
import math
import jax.numpy as jnp
from flax.struct import dataclass


class NoiseScheduleVP:
    """
    The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
    We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
    Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:
        log_alpha_t = self.marginal_log_mean_coeff(t)
        sigma_t = self.marginal_std(t)
        lambda_t = self.marginal_lambda(t)
    Moreover, as lambda(t) is an invertible function, we also support its inverse function:
        t = self.inverse_lambda(lambda_t)
    """

    @abstractmethod
    def marginal_log_mean_coeff(self, t):
        pass

    def marginal_std(self, t):
        return jnp.sqrt(1. - jnp.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * jnp.log(1. - jnp.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    @abstractmethod
    def inverse_lambda(self, lamb):
        pass

    def q_sample(self, x, t, eps):
        alpha = jnp.exp(self.marginal_log_mean_coeff(t))
        sigma = self.marginal_std(t)
        return alpha * x + sigma * eps


@dataclass
class LinearNoiseScheduleVP(NoiseScheduleVP):
    """ 
    Linear (continuous-time) noise schedule used for inference

    :param beta_0: A `float` number. The smallest beta for the linear schedule.
    :param beta_1: A `float` number. The largest beta for the linear schedule.
    :param T: A `float` number. The ending time of the forward process.
    """

    beta_0: float = 0.1
    beta_1: float = 20
    T: float = 1

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        tmp = 2. * (self.beta_1 - self.beta_0) * jnp.logaddexp(-2. * lamb, jnp.zeros((1,)).to(lamb))
        Delta = self.beta_0**2 + tmp
        return tmp / (jnp.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)


@dataclass
class CosineNoiseScheduleVP(NoiseScheduleVP):
    """ Cosine (continuous-time) noise schedule used for training

    :param s: A `float` number. The hyperparameter in the cosine schedule.
    :param beta_max: A `float` number. The hyperparameter in the cosine schedule.
    :param T: A `float` number. The ending time of the forward process.
    """
    s: float = 0.008
    beta_max: float = 999.
    log_alpha_0: float = math.log(math.cos(s / (1. + s) * math.pi / 2.))
    T: float = 0.9946

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        log_alpha_t = jnp.log(jnp.cos((t + self.s) / (1. + self.s) * math.pi / 2.)) - self.log_alpha_0
        return log_alpha_t

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        log_alpha = -0.5 * jnp.logaddexp(-2. * lamb, 0.)
        t = jnp.arccos(jnp.exp(log_alpha + self.log_alpha_0)) * 2. * (1. + self.s) / math.pi - self.s
        return t