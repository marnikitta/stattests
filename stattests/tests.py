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


def get_smoothed_ctrs(clicks_0, views_0, clicks_1, views_1, smothing_factor=200.):
    """
    Calculates smoothed ctr for every user in every experiment both in treatment and control groups
    Smoothed_ctr = (user_clicks + smothing_factor * global_ctr) / (user_views + smothing_factor)
    :param clicks_0: np.array shape (n_experiments, n_users), clicks of every user from control group in every experiment
    :param views_0: np.array shape (n_experiments, n_users), views of every user from control group in every experiment
    :param clicks_1: np.array shape (n_experiments, n_users), clicks of every user from treatment group in every experiment
    :param views_1: np.array shape (n_experiments, n_users), views of every user from treatment group in every experiment
    :param smothing_factor: float
    :return: (np.array, np.array) shape (n_experiments, n_users), smoothed ctrs for every user in every experiment
    """
    global_ctr = (np.sum(clicks_0, axis=1) / np.sum(views_0, axis=1)).reshape(-1, 1)
    ctrs_0 = (clicks_0 + smothing_factor * global_ctr) / (views_0 + smothing_factor)
    ctrs_1 = (clicks_1 + smothing_factor * global_ctr) / (views_1 + smothing_factor)
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


def bucketization(ctrs_0, weights_0, ctrs_1, weights_1, n_buckets=200):
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


def binomial_test(global_ctr_0, total_views_0, global_ctr_1, total_views_1):
    """
    Calculates two-sided p-values for all the experiments on global CTRs using z-test
    :param global_ctr_0: np.array shape (n_experiments), global ctr in control group in every experiment
    :param total_views_0: np.array shape (n_experiments), sum of views in control group in every experiment
    :param global_ctr_1: np.array shape (n_experiments), global ctr in treatment group in every experiment
    :param total_views_1: np.array shape (n_experiments), sum of views in treatment group in every experiment
    :return: np.array shape (n_experiments), two-sided p-values of delta-method on CTRs in all the experimetns
    """
    overall_ctrs = (global_ctr_0 * total_views_0 + global_ctr_1 * total_views_1) / (total_views_0 + total_views_1)
    z_stats = (global_ctr_0 - global_ctr_1) / np.sqrt(
        overall_ctrs * (1 - overall_ctrs) * (1. / total_views_0 + 1. / total_views_1))
    return 2 * np.minimum(scipy.stats.norm(0, 1).cdf(z_stats), 1 - scipy.stats.norm(0, 1).cdf(z_stats))


def delta_method_ctrs(clicks_0, views_0, clicks_1, views_1):
    """
    Calculates two-sided p-values for all the experiments on CTRs using delta-method
    :param clicks_0: np.array shape (n_experiments, n_users), clicks of every user from control group in every experiment
    :param views_0: np.array shape (n_experiments, n_users), views of every user from control group in every experiment
    :param clicks_1: np.array shape (n_experiments, n_users), clicks of every user from treatment group in every experiment
    :param views_1: np.array shape (n_experiments, n_users), views of every user from treatment group in every experiment
    :return: np.array shape (n_experiments), two-sided p-values of delta-method on CTRs in all the experimetns
    """
    n_experiments, n_users = views_0.shape

    mean_clicks_0, var_clicks_0 = np.mean(clicks_0, axis=1), np.var(clicks_0, axis=1)
    mean_clicks_1, var_clicks_1 = np.mean(clicks_1, axis=1), np.var(clicks_1, axis=1)

    mean_views_0, var_views_0 = np.mean(views_0, axis=1), np.var(views_0, axis=1)
    mean_views_1, var_views_1 = np.mean(views_1, axis=1), np.var(views_1, axis=1)

    cov_0 = np.mean((clicks_0 - mean_clicks_0.reshape(-1, 1)) * (views_0 - mean_views_0.reshape(-1, 1)),
                    axis=1)
    cov_1 = np.mean((clicks_1 - mean_clicks_1.reshape(-1, 1)) * (views_1 - mean_views_1.reshape(-1, 1)),
                    axis=1)

    var_0 = var_clicks_0 / mean_views_0 ** 2 + var_views_0 * mean_clicks_0 ** 2 / mean_views_0 ** 4 - 2 * mean_clicks_0 / mean_views_0 ** 3 * cov_0
    var_1 = var_clicks_1 / mean_views_1 ** 2 + var_views_1 * mean_clicks_1 ** 2 / mean_views_1 ** 4 - 2 * mean_clicks_1 / mean_views_1 ** 3 * cov_1

    ctrs_0 = np.sum(clicks_0, axis=1) / np.sum(views_0, axis=1)
    ctrs_1 = np.sum(clicks_1, axis=1) / np.sum(views_1, axis=1)

    z_stats = (ctrs_1 - ctrs_0) / np.sqrt(var_0 / n_users + var_1 / n_users)
    p_ctr_delta = 2 * np.minimum(scipy.stats.norm(0, 1).cdf(z_stats), 1 - scipy.stats.norm(0, 1).cdf(z_stats))
    return p_ctr_delta


def intra_user_correlation_aware_weights(clicks_0, views_0, views_1):
    """
    Calculates weights for UMVUE global ctr estimate for every user in every experiment both in treatment and control groups
    :param clicks_0: np.array shape (n_experiments, n_users), clicks of every user from control group in every experiment
    :param views_0: np.array shape (n_experiments, n_users), views of every user from control group in every experiment
    :param views_1: np.array shape (n_experiments, n_users), views of every user from treatment group in every experiment
    :return: (np.array, np.array) shape (n_experiments, n_users), weights for every user in every experiment
    """
    ri = clicks_0 / views_0
    s3 = clicks_0 * (1 - ri) ** 2 + (views_0 - clicks_0) * ri ** 2
    s3 = np.sum(s3, axis=1) / np.sum(views_0 - 1, axis=1)

    rb = np.mean(clicks_0 / views_0, axis=1).reshape(-1, 1)
    s2 = clicks_0 * (1 - rb) ** 2 + (views_0 - clicks_0) * rb ** 2
    s2 = np.sum(s2, axis=1) / (np.sum(views_0, axis=1) - 1)

    rho = np.maximum(0, 1 - s3 / s2).reshape(-1, 1)

    w_0 = views_0 / (1 + (views_0 - 1) * rho)
    w_1 = views_1 / (1 + (views_1 - 1) * rho)
    return w_0, w_1


def linearization_of_clicks(clicks_0, views_0, clicks_1, views_1):
    """
    Fits linear model clicks = k * views and returns clicks - k * views (e.g. it accounts for correlation of
    clicks and views)
    :param clicks_0: np.array shape (n_experiments, n_users), clicks of every user from control group in every experiment
    :param views_0: np.array shape (n_experiments, n_users), views of every user from control group in every experiment
    :param clicks_1: np.array shape (n_experiments, n_users), clicks of every user from treatment group in every experiment
    :param views_1: np.array shape (n_experiments, n_users), views of every user from treatment group in every experiment
    :return: (np.array, np_array) shape (n_experiments), linearized clicks for every user in every experiment
    """
    k = clicks_0.flatten().sum() / views_0.flatten().sum()
    L_0 = clicks_0 - k * views_0
    L_1 = clicks_1 - k * views_1
    return L_0, L_1


def permutation_test(clicks_0: np.ndarray,
                     views_0: np.ndarray,
                     clicks_1: np.ndarray,
                     views_1: np.ndarray,
                     samples: int = 2000) -> np.ndarray:
    n_experiments = views_0.shape[0]
    n_users_0 = views_0.shape[1]
    n_users_1 = views_1.shape[1]

    permutations = np.zeros((samples, n_users_0 + n_users_1)).astype(np.int32)
    permutation = np.arange(n_users_0 + n_users_1)
    for i in range(samples):
        np.random.shuffle(permutation)
        permutations[i] = permutation.copy()
    permutation_flags = (permutations < n_users_0).astype(np.int32)

    concated_views = np.hstack((views_0, views_1))
    concated_clicks = np.hstack((clicks_0, clicks_1))

    clicks_sum_0 = np.matmul(concated_clicks, permutation_flags.T)
    clicks_sum_1 = np.matmul(concated_clicks, 1 - permutation_flags.T)

    views_sum_0 = np.matmul(concated_views, permutation_flags.T)
    views_sum_1 = np.matmul(concated_views, 1 - permutation_flags.T)

    null_stats = clicks_sum_1 / views_sum_1 - clicks_sum_0 / views_sum_0
    null_stats = np.sort(null_stats)
    p_values = np.zeros(n_experiments)

    for i in range(n_experiments):
        exp_stat = clicks_1[i].sum() / views_1[i].sum() - clicks_0[i].sum() / views_0[i].sum()
        insert_position = np.searchsorted(null_stats[i], exp_stat)
        p_values[i] = 2 * np.minimum(samples - insert_position, insert_position) / samples

    return p_values
