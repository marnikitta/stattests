from typing import Tuple

import numpy as np
import scipy.stats


def generate_data(skew: float = 2.0,
                  N: int = 5000,
                  NN: int = 2000,
                  success_rate: float = 0.02,
                  uplift: float = 0.1,
                  beta: float = 250.) -> Tuple[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Generates experimental data for N users in NN experiments
    :param skew: float, skewness of attempts distribution
    :param N: int, number of users in each experimental group (in control and in treatment)
    :param NN: int, number of experiments
    :param success_rate: float, mean success rate in control group
    :param uplift: float, relative uplift of mean success rate in treatment group
    :param beta: float, parameter of success rate distribution
    :return: (np.array, np.array, np.array, np.array, np.array) shape (NN, N), attempts in control group,
    successes in control group, attempts in treatment group, successes in treatment group, ground-truth success_rates for control group
    """
    attempts_0 = np.exp(scipy.stats.norm(1, skew).rvs(NN * N)).astype(np.int).reshape(NN, N) + 1
    attempts_1 = np.exp(scipy.stats.norm(1, skew).rvs(NN * N)).astype(np.int).reshape(NN, N) + 1

    # attempts is always positive, abs is fixing numerical issues with high skewness
    attempts_0 = np.absolute(attempts_0)
    attempts_1 = np.absolute(attempts_1)

    alpha_0 = success_rate * beta / (1 - success_rate)
    success_rate_0 = scipy.stats.beta(alpha_0, beta).rvs(NN * N).reshape(NN, N)

    alpha_1 = success_rate * (1 + uplift) * beta / (1 - success_rate * (1 + uplift))
    success_rate_1 = scipy.stats.beta(alpha_1, beta).rvs(NN * N).reshape(NN, N)

    successes_0 = scipy.stats.binom(n=attempts_0, p=success_rate_0).rvs()
    successes_1 = scipy.stats.binom(n=attempts_1, p=success_rate_1).rvs()
    return ((attempts_0.astype(np.float64), successes_0.astype(np.float64)),
            (attempts_1.astype(np.float64), successes_1.astype(np.float64)),
            success_rate_0.astype(np.float64))
