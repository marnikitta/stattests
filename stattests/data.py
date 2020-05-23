import pathlib
from typing import Callable, Tuple

import numpy as np

from stattests.generation import generate_data
from stattests.tests import t_test, mannwhitney, delta_method_ctrs, bootstrap, linearization_of_clicks, bucketization, \
    intra_user_correlation_aware_weights, get_smoothed_ctrs, binomial_test


def wpv(data_dir: str,
        codename: str,
        ab_data_callback: Callable[[], np.ndarray],
        aa_data_callback: Callable[[], np.ndarray],
        NN: int,
        N: int,
        uplift: float,
        success_rate: float,
        beta: float,
        skew: float):
    filename = f'{data_dir}/NN={NN}/N={N}/uplift={uplift}/success_rate={success_rate}/beta={beta}/skew={skew}/{codename}'
    data_path = pathlib.Path(filename)
    if data_path.exists():
        return

    ab_data = ab_data_callback()
    aa_data = aa_data_callback()
    data_path.mkdir(parents=True, exist_ok=True)

    with (data_path / 'ab_data').open('w') as f:
        f.write(','.join(map(str, ab_data)))
    with (data_path / 'aa_data').open('w') as f:
        f.write(','.join(map(str, aa_data)))


def rpv(data_dir: str,
        codename: str,
        NN: int,
        N: int,
        uplift: float,
        success_rate: float,
        beta: float,
        skew: float) -> Tuple[np.ndarray, np.ndarray]:
    filename = f'{data_dir}/NN={NN}/N={N}/uplift={uplift}/success_rate={success_rate}/beta={beta}/skew={skew}/{codename}'
    data_path = pathlib.Path(filename)
    with (data_path / 'ab_data').open('r') as f:
        line = f.readline()
        ab_data = np.array(list(map(float, line.split(','))))
    with (data_path / 'aa_data').open('r') as f:
        line = f.readline()
        aa_data = np.array(list(map(float, line.split(','))))

    return ab_data, aa_data


def apply_all_tests(data_dir: str,
                    NN: int,
                    N: int,
                    uplift: float,
                    success_rate: float,
                    beta: float,
                    skew: float):
    ab_params = {'success_rate': success_rate, 'uplift': uplift, 'beta': beta, 'skew': skew, 'N': N, 'NN': NN}
    aa_params = {'success_rate': success_rate, 'uplift': 0.0, 'beta': beta, 'skew': skew, 'N': N, 'NN': NN}

    (views_0_ab, clicks_0_ab), (views_1_ab, clicks_1_ab), gt_success_rates = generate_data(**ab_params)
    (views_0_aa, clicks_0_aa), (views_1_aa, clicks_1_aa), _ = generate_data(**aa_params)

    wpv(data_dir, 'ttest_successes_count', lambda: t_test(clicks_0_ab, clicks_1_ab),
        lambda: t_test(clicks_0_aa, clicks_1_aa),
        **ab_params)
    wpv(data_dir, 'mannwhitney_successes_count', lambda: mannwhitney(clicks_0_ab, clicks_1_ab),
        lambda: mannwhitney(clicks_0_aa, clicks_1_aa), **ab_params)

    wpv(data_dir, 'delta', lambda: delta_method_ctrs(clicks_0_ab, views_0_ab, clicks_1_ab, views_1_ab),
        lambda: delta_method_ctrs(clicks_0_aa, views_0_aa, clicks_1_aa, views_1_aa), **ab_params)

    wpv(data_dir, 'bootstrap',
        lambda: bootstrap(clicks_0_ab / views_0_ab, views_0_ab, clicks_1_ab / views_1_ab, views_1_ab),
        lambda: bootstrap(clicks_0_aa / views_0_aa, views_0_aa, clicks_1_aa / views_1_aa, views_1_aa),
        **ab_params)

    linearized_0_ab, linearized_1_ab = linearization_of_clicks(clicks_0_ab, views_0_ab, clicks_1_ab,
                                                               views_1_ab)
    linearized_0_aa, linearized_1_aa = linearization_of_clicks(clicks_0_aa, views_0_aa, clicks_1_aa,
                                                               views_1_aa)
    wpv(data_dir, 'linearization', lambda: t_test(linearized_0_ab, linearized_1_ab),
        lambda: t_test(linearized_0_aa, linearized_1_aa),
        **ab_params)

    wpv(data_dir, 'buckets', lambda: bucketization(clicks_0_ab / views_0_ab, np.ones(shape=views_0_ab.shape),
                                                   clicks_1_ab / views_1_ab, np.ones(shape=views_1_ab.shape)),
        lambda: bucketization(clicks_0_aa / views_0_aa, np.ones(shape=views_0_aa.shape),
                              clicks_1_aa / views_1_aa, np.ones(shape=views_1_aa.shape)), **ab_params)

    wpv(data_dir, 'buckets_ctrs', lambda: bucketization(clicks_0_ab / views_0_ab, views_0_ab,
                                                        clicks_1_ab / views_1_ab, views_1_ab),
        lambda: bucketization(clicks_0_aa / views_0_aa, views_0_aa,
                              clicks_1_aa / views_1_aa, views_0_aa), **ab_params)

    wpv(data_dir, 't_test_ctrs', lambda: t_test(clicks_0_ab / views_0_ab, clicks_1_ab / views_1_ab),
        lambda: t_test(clicks_0_aa / views_0_aa, clicks_1_aa / views_1_aa), **ab_params)

    corr_aware_w_0_ab, corr_aware_w_1_ab = intra_user_correlation_aware_weights(clicks_0_ab, views_0_ab,
                                                                                views_1_ab)
    corr_aware_w_0_aa, corr_aware_w_1_aa = intra_user_correlation_aware_weights(clicks_0_aa, views_0_aa,
                                                                                views_1_aa)

    wpv(data_dir, 'weighted_bootstrap',
        lambda: bootstrap(clicks_0_ab / views_0_ab, corr_aware_w_0_ab, clicks_1_ab / views_1_ab,
                          corr_aware_w_1_ab),
        lambda: bootstrap(clicks_0_aa / views_0_aa, corr_aware_w_0_aa, clicks_1_aa / views_1_aa,
                          corr_aware_w_1_aa),
        **ab_params)

    wpv(data_dir, 'weighted_linearization',
        lambda: t_test(linearized_0_ab * corr_aware_w_0_ab, linearized_1_ab * corr_aware_w_1_ab),
        lambda: t_test(linearized_0_aa * corr_aware_w_0_aa, linearized_1_aa * corr_aware_w_1_aa), **ab_params)

    wpv(data_dir, 'weighted_t_test_ctrs',
        lambda: t_test(clicks_0_ab / views_0_ab * corr_aware_w_0_ab,
                       clicks_1_ab / views_1_ab * corr_aware_w_1_ab),
        lambda: t_test(clicks_0_aa / views_0_aa * corr_aware_w_0_aa,
                       clicks_1_aa / views_1_aa * corr_aware_w_1_aa),
        **ab_params)

    wpv(data_dir, 'weighted_buckets',
        lambda: bucketization(clicks_0_ab / views_0_ab, corr_aware_w_0_ab, clicks_1_ab / views_1_ab,
                              corr_aware_w_1_ab),
        lambda: bucketization(clicks_0_aa / views_0_aa, corr_aware_w_0_aa, clicks_1_aa / views_1_aa,
                              corr_aware_w_1_aa),
        **ab_params)

    wpv(data_dir, 'weighted_sqr_bootstrap',
        lambda: bootstrap(clicks_0_ab / views_0_ab, np.sqrt(views_0_ab), clicks_1_ab / views_1_ab,
                          np.sqrt(views_1_ab)),
        lambda: bootstrap(clicks_0_aa / views_0_aa, np.sqrt(views_0_aa), clicks_1_aa / views_1_aa,
                          np.sqrt(views_1_aa)),
        **ab_params)

    wpv(data_dir, 'weighted_sqr_linearization',
        lambda: t_test(linearized_0_ab * np.sqrt(views_0_ab), linearized_1_ab * np.sqrt(views_1_ab)),
        lambda: t_test(linearized_0_aa * np.sqrt(views_0_aa), linearized_1_aa * np.sqrt(views_1_aa)),
        **ab_params)

    wpv(data_dir, 'weighted_sqr_t_test_ctrs',
        lambda: t_test(clicks_0_ab / views_0_ab * np.sqrt(views_0_ab),
                       clicks_1_ab / views_1_ab * np.sqrt(views_1_ab)),
        lambda: t_test(clicks_0_aa / views_0_aa * np.sqrt(views_0_aa),
                       clicks_1_aa / views_1_aa * np.sqrt(views_1_aa)),
        **ab_params)

    wpv(data_dir, 'weighted_sqr_buckets',
        lambda: bucketization(clicks_0_ab / views_0_ab, np.sqrt(views_0_ab), clicks_1_ab / views_1_ab,
                              np.sqrt(views_1_ab)),
        lambda: bucketization(clicks_0_aa / views_0_aa, np.sqrt(views_0_aa), clicks_1_aa / views_1_aa,
                              np.sqrt(views_1_aa)),
        **ab_params)

    wpv(data_dir, 'weighted_sqr_buckets',
        lambda: bucketization(clicks_0_ab / views_0_ab, np.sqrt(views_0_ab), clicks_1_ab / views_1_ab,
                              np.sqrt(views_1_ab)),
        lambda: bucketization(clicks_0_aa / views_0_aa, np.sqrt(views_0_aa), clicks_1_aa / views_1_aa,
                              np.sqrt(views_1_aa)),
        **ab_params)

    smoothed_ctrs_0_ab, smoothed_ctrs_1_ab = get_smoothed_ctrs(clicks_0_ab, views_0_ab,
                                                               clicks_1_ab, views_1_ab)
    smoothed_ctrs_0_aa, smoothed_ctrs_1_aa = get_smoothed_ctrs(clicks_0_aa, views_0_aa,
                                                               clicks_1_aa, views_1_aa)
    wpv(data_dir, 'ttest_smoothed',
        lambda: t_test(smoothed_ctrs_0_ab, smoothed_ctrs_1_ab),
        lambda: t_test(smoothed_ctrs_0_aa, smoothed_ctrs_1_aa),
        **ab_params)

    global_ctr_0_ab = clicks_0_ab.sum(axis=1) / views_0_ab.sum(axis=1)
    global_ctr_1_ab = clicks_1_ab.sum(axis=1) / views_1_ab.sum(axis=1)
    global_ctr_0_aa = clicks_0_aa.sum(axis=1) / views_0_aa.sum(axis=1)
    global_ctr_1_aa = clicks_1_aa.sum(axis=1) / views_1_aa.sum(axis=1)

    wpv(data_dir, 'binomial_test',
        lambda: binomial_test(global_ctr_0_ab, views_0_ab.sum(axis=1),
                              global_ctr_1_ab, views_1_ab.sum(axis=1)),
        lambda: binomial_test(global_ctr_0_aa, views_0_aa.sum(axis=1),
                              global_ctr_1_aa, views_1_aa.sum(axis=1)),
        **ab_params)
