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
    :param skew: float, skewness of views distribution
    :param N: int, number of users in each experimental group (in control and in treatment)
    :param NN: int, number of experiments
    :param success_rate: float, mean success rate in control group
    :param uplift: float, relative uplift of mean success rate in treatment group
    :param beta: float, parameter of success rate distribution
    :return: (np.array, np.array, np.array, np.array, np.array) shape (NN, N), views in control group,
    clicks in control group, views in treatment group, clicks in treatment group, ground truth user CTRs for control group
    """
    views_0 = np.exp(scipy.stats.norm(1, skew).rvs(NN * N)).astype(np.int).reshape(NN, N) + 1
    views_1 = np.exp(scipy.stats.norm(1, skew).rvs(NN * N)).astype(np.int).reshape(NN, N) + 1

    # views are always positive, abs is fixing numerical issues with high skewness
    views_0 = np.absolute(views_0)
    views_1 = np.absolute(views_1)

    alpha_0 = success_rate * beta / (1 - success_rate)
    success_rate_0 = scipy.stats.beta(alpha_0, beta).rvs(NN * N).reshape(NN, N)

    alpha_1 = success_rate * (1 + uplift) * beta / (1 - success_rate * (1 + uplift))
    success_rate_1 = scipy.stats.beta(alpha_1, beta).rvs(NN * N).reshape(NN, N)

    clicks_0 = scipy.stats.binom(n=views_0, p=success_rate_0).rvs()
    clicks_1 = scipy.stats.binom(n=views_1, p=success_rate_1).rvs()
    return ((views_0.astype(np.float64), clicks_0.astype(np.float64)),
            (views_1.astype(np.float64), clicks_1.astype(np.float64)),
            success_rate_0.astype(np.float64))
