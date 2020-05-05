import numpy as np
import scipy.stats


def t_test(a, b):
    """
    Calculates two-sided t-test p-values for multiple experiments
    :param a: np.array shape (n_experiments, n_users), metric values in control group
    :param b: np.array shape (n_experiments, n_users), metric values in treatment group
    :return: np.array shape (n_experiments), two-sided p-values of t-test in all experimetns
    """
    result = list(map(lambda x: scipy.stats.ttest_ind(x[0], x[1]).pvalue, zip(a, b)))
    return np.array(result)


def mannwhitney(a, b):
    """
    Calculates two-sided t-test p-values for multiple experiments
    :param a: np.array shape (n_experiments, n_users), metric values in control group
    :param b: np.array shape (n_experiments, n_users), metric values in treatment group
    :return: np.array shape (n_experiments), two-sided p-values of Mann-Whitney test in all experimetns
    """
    result = list(map(lambda x: scipy.stats.mannwhitneyu(x[0], x[1], alternative='two-sided').pvalue, zip(a, b)))
    return np.array(result)


def get_smoothed_ctrs(successes_0, attempts_0, successes_1, attempts_1, smothing_factor=200.):
    """
    Calculates smoothed ctr for every user in every experiment both in treatment and control groups
    Smoothed_ctr = (user_successes + smothing_factor * global_ctr) / (user_attempts + smothing_factor)
    :param successes_0: np.array shape (n_experiments, n_users), successes of every user from control group in every experiment
    :param attempts_0: np.array shape (n_experiments, n_users), attempts of every user from control ggroup in every experiment
    :param successes_1: np.array shape (n_experiments, n_users), successes of every user from treatment group in every experiment
    :param attempts_1: np.array shape (n_experiments, n_users), attempts of every user from treatment ggroup in every experiment
    :param smothing_factor: float
    :return: (np.array, np.array) shape (n_experiments, n_users), smoothed ctrs for every user in every experiment
    """
    global_ctr = (np.sum(successes_0, axis=1) / np.sum(attempts_0, axis=1)).reshape(-1, 1)
    ctrs_0 = (successes_0 + smothing_factor * global_ctr) / (attempts_0 + smothing_factor)
    ctrs_1 = (successes_1 + smothing_factor * global_ctr) / (attempts_1 + smothing_factor)
    return ctrs_0, ctrs_1


def bootstrap(ctrs_0, weights_0, ctrs_1, weights_1, n_bootstrap=2000):
    """
    Does weighted bootstrap and calculates p-value according to the bootstraped distribution
    :param ctrs_0: np.array shape (n_experiments, n_users), CTRs of every user from control group in every experiment
    :param weights_0: np.array (n_experiments, n_users), weight of every user from control group in every experiment
    :param ctrs_1: np.array (n_experiments, n_users), CTRs of every user from treatment group in every experiment
    :param weights_1: np.array (n_experiments, n_users), weight of every user from treatment group in every experiment
    :param n_bootstrap: int - for every experiment wi will generate n_bootstrap bootstrap pseudo-samples
    :return: np.array shape (n_experiments), two-sided p-values of weighted bootstrap test in all experimetns
    """
    poisson_bootstraps = scipy.stats.poisson(1).rvs((n_bootstrap, ctrs_0.shape[1])).astype(np.int64)

    values_0 = np.matmul(ctrs_0 * weights_0, poisson_bootstraps.T)
    weights_0 = np.matmul(weights_0, poisson_bootstraps.T)

    values_1 = np.matmul(ctrs_1 * weights_1, poisson_bootstraps.T)
    weights_1 = np.matmul(weights_1, poisson_bootstraps.T)

    deltas = values_1 / weights_1 - values_0 / weights_0

    positions = np.zeros(deltas.shape[0])
    deltas = np.sort(deltas, axis=1)
    for n in np.arange(deltas.shape[0]):
        positions[n] = np.searchsorted(deltas[n], 0)

    return 2 * np.minimum(positions, n_bootstrap - positions) / n_bootstrap


def buckets(ctrs_0, weights_0, ctrs_1, weights_1, n_buckets=50):
    """
    Does weighted bucketization and calculates p-values for all experiments using t_test
    :param ctrs_0: np.array shape (n_experiments, n_users), CTRs of every user from control group in every experiment
    :param weights_0: np.array (n_experiments, n_users), weight of every user from control group in every experiment
    :param ctrs_1: np.array (n_experiments, n_users), CTRs of every user from treatment group in every experiment
    :param weights_1: np.array (n_experiments, n_users), weight of every user from treatment group in every experiment
    :param n_buckets: int, nubmer of buckets
    :return: np.array shape (n_experiments), two-sided p-values of weighted bucketization test in all the experimetns
    """

    n_experiments, n_users = ctrs_0.shape

    values_0 = np.zeros((n_experiments, n_buckets))
    values_1 = np.zeros((n_experiments, n_buckets))

    for b in np.arange(n_buckets):
        ind = np.arange(b * n_users / n_buckets, b * n_users / n_buckets + n_users / n_buckets).astype(np.int)
        values_0[:, b] = np.sum(ctrs_0[:, ind] * weights_0[:, ind], axis=1) / np.sum(weights_0[:, ind], axis=1)
        values_1[:, b] = np.sum(ctrs_1[:, ind] * weights_1[:, ind], axis=1) / np.sum(weights_1[:, ind], axis=1)

    return t_test(values_0, values_1)


def proportions_diff_z_test(success_rate_0, attempts_0, success_rate_1, attempts_1):
    """
    Calculates two-sided p-values for all the experiments on global CTRs using z-test
    :param success_rate_0: np.array shape (n_experiments), global ctr in control group in every experiment
    :param attempts_0: np.array shape (n_experiments), sum of attempts in control group in every experiment
    :param success_rate_1: np.array shape (n_experiments), global ctr in treatment group in every experiment
    :param attempts_1: np.array shape (n_experiments), sum of attempts in treatment ggroup in every experiment
    :return: np.array shape (n_experiments), two-sided p-values of delta-method on CTRs in all the experimetns
    """
    overall_ctrs = (success_rate_0 * attempts_0 + success_rate_1 * attempts_1) / (attempts_0 + attempts_1)
    z_stats = (success_rate_0 - success_rate_1) / np.sqrt(overall_ctrs * (1 - overall_ctrs) * (1. / attempts_0 + 1. / attempts_1))
    return 2 * np.minimum(scipy.stats.norm(0, 1).cdf(z_stats), 1 - scipy.stats.norm(0, 1).cdf(z_stats))


def delta_method_ctrs(successes_0, attempts_0, successes_1, attempts_1):
    """
    Calculates two-sided p-values for all the experiments on CTRs using delta-method
    :param successes_0: np.array shape (n_experiments, n_users), successes of every user from control group in every experiment
    :param attempts_0: np.array shape (n_experiments, n_users), attempts of every user from control ggroup in every experiment
    :param successes_1: np.array shape (n_experiments, n_users), successes of every user from treatment group in every experiment
    :param attempts_1: np.array shape (n_experiments, n_users), attempts of every user from treatment ggroup in every experiment
    :return: np.array shape (n_experiments), two-sided p-values of delta-method on CTRs in all the experimetns
    """
    n_experiments, n_users = attempts_0.shape

    mean_successes_0, var_successes_0 = np.mean(successes_0, axis=1), np.var(successes_0, axis=1)
    mean_successes_1, var_successes_1 = np.mean(successes_1, axis=1), np.var(successes_1, axis=1)

    mean_attempts_0, var_attempts_0 = np.mean(attempts_0, axis=1), np.var(attempts_0, axis=1)
    mean_attempts_1, var_attempts_1 = np.mean(attempts_1, axis=1), np.var(attempts_1, axis=1)

    cov_0 = np.mean((successes_0 - mean_successes_0.reshape(-1, 1)) * (attempts_0 - mean_attempts_0.reshape(-1, 1)), axis=1)
    cov_1 = np.mean((successes_1 - mean_successes_1.reshape(-1, 1)) * (attempts_1 - mean_attempts_1.reshape(-1, 1)), axis=1)

    var_0 = var_successes_0 / mean_attempts_0 ** 2 + var_attempts_0 * mean_successes_0 ** 2 / mean_attempts_0 ** 4 - 2 * mean_successes_0 / mean_attempts_0 ** 3 * cov_0
    var_1 = var_successes_1 / mean_attempts_1 ** 2 + var_attempts_1 * mean_successes_1 ** 2 / mean_attempts_1 ** 4 - 2 * mean_successes_1 / mean_attempts_1 ** 3 * cov_1

    ctrs_0 = np.sum(successes_0, axis=1) / np.sum(attempts_0, axis=1)
    ctrs_1 = np.sum(successes_1, axis=1) / np.sum(attempts_1, axis=1)

    z_stats = (ctrs_1 - ctrs_0) / np.sqrt(var_0 / n_users + var_1 / n_users)
    p_ctr_delta = 2 * np.minimum(scipy.stats.norm(0, 1).cdf(z_stats), 1 - scipy.stats.norm(0, 1).cdf(z_stats))
    return p_ctr_delta


def intra_user_correlation_aware_weights(successes_0, attempts_0, attempts_1):
    """
    Calculates weights for UMVUE global ctr estimate for every user in every experiment both in treatment and control groups
    :param successes_0: np.array shape (n_experiments, n_users), successes of every user from control group in every experiment
    :param attempts_0: np.array shape (n_experiments, n_users), attempts of every user from control ggroup in every experiment
    :param attempts_1: np.array shape (n_experiments, n_users), attempts of every user from treatment ggroup in every experiment
    :return: (np.array, np.array) shape (n_experiments, n_users), weights for every user in every experiment
    """
    ri = successes_0 / attempts_0
    s3 = successes_0 * (1 - ri) ** 2 + (attempts_0 - successes_0) * ri ** 2
    s3 = np.sum(s3, axis=1) / np.sum(attempts_0 - 1, axis=1)

    rb = np.mean(successes_0 / attempts_0, axis=1).reshape(-1, 1)
    s2 = successes_0 * (1 - rb) ** 2 + (attempts_0 - successes_0) * rb ** 2
    s2 = np.sum(s2, axis=1) / (np.sum(attempts_0, axis=1) - 1)

    rho = np.maximum(0, 1 - s3 / s2).reshape(-1, 1)

    w_0 = attempts_0 / (1 + (attempts_0 - 1) * rho)
    w_1 = attempts_1 / (1 + (attempts_1 - 1) * rho)
    return w_0, w_1

def linearization_of_successes(successes_0, attempts_0, successes_1, attempts_1):
    """
    Fits linear model successes = k * attempts and returns successes - k * attempts (e.g. it accounts for correlation of
    successes and attempts)
    :param successes_0: np.array shape (n_experiments, n_users), successes of every user from control group in every experiment
    :param attempts_0: np.array shape (n_experiments, n_users), attempts of every user from control ggroup in every experiment
    :param successes_1: np.array shape (n_experiments, n_users), successes of every user from treatment group in every experiment
    :param attempts_1: np.array shape (n_experiments, n_users), attempts of every user from treatment ggroup in every experiment
    :return: (np.array, np_array) shape (n_experiments), linearized successes for every user in every experiment
    """
    k = np.mean(successes_0 / attempts_0, axis=1).reshape(-1, 1)
    k = successes_0.flatten().sum() / attempts_0.flatten().sum()
    L_0 = successes_0 - k * attempts_0
    L_1 = successes_1 - k * attempts_1
    return L_0, L_1