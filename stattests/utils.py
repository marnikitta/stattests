import pathlib
from typing import Tuple, Dict, Callable, Optional, Set

import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

from stattests.generation import generate_data
from stattests.tests import t_test, mannwhitney, delta_method_ctrs, linearization_of_successes, bootstrap, buckets, \
    intra_user_correlation_aware_weights, get_smoothed_ctrs

codenames2titles = {'ttest_successes_count': ('T-test, successes count', 'r--'),
                    'mannwhitney_successes_count': ('Mann-Whitney test, successes count', 'b--'),
                    'delta': ('Delta-method, global CTR', 'g--'),
                    'bootstrap': ('Bootstrap, global CTR', 'k--'),
                    'linearization': ('Linearization of successes', 'm--'),
                    'buckets': ('Bucketization, bucket CTR', 'c--'),
                    't_test_ctrs': ('T-test, user-CTR', 'y--'),
                    'weighted_bootstrap': ('Weighted bootstrap', 'k:'),
                    'weighted_linearization': ('Weighted linearization', 'm:'),
                    'weighted_buckets': ('Weighted bucketization', 'c:'),
                    'weighted_t_test_ctrs': ('Weighted t-test CTRs', 'y:'),
                    'weighted_sqr_bootstrap': ('Weighted sqr bootstrap', 'k-.'),
                    'weighted_sqr_linearization': ('Weighted sqr linearization', 'm-.'),
                    'weighted_sqr_buckets': ('Weighted sqr bucketization', 'c-.'),
                    'weighted_sqr_t_test_ctrs': ('Weighted sqr t-test CTRs', 'y-.'),
                    'ttest_smoothed': ('T-test, smoothed CTRs', 'r-.')}


def plot_cdf(data, label, ax, linetype):
    sorted_data = np.sort(data)
    position = scipy.stats.rankdata(sorted_data, method='ordinal')
    cdf = position / data.shape[0]

    sorted_data = np.hstack((sorted_data, 1))
    cdf = np.hstack((cdf, 1))

    return ax.plot(sorted_data, cdf, linetype, label=label, linewidth=1.5)


def plot_summary(dict2plot: Dict[str, Tuple[np.ndarray, np.ndarray, str]],
                 attempts_0: np.ndarray,
                 ground_truth_success_rates: np.ndarray, fix_axis: bool = True):
    """
    :param dict2plot: dict[str, (np.array, str)]
    :param ground_truth_success_rates: np.array
    :param attempts_0: np.array
    :return: 
    """
    fig, ((ax_h1, ax_legend, ax_h0), (ax_powers, ax_attempts, ax_successes)) = plt.subplots(2, 3, figsize=(20, 10),
                                                                                            dpi=200)

    fig.subplots_adjust(wspace=0.2, hspace=0.3)
    # fig.set_tight_layout(True)

    ax_h1.plot(np.linspace(0, 1, 10000), np.linspace(0, 1, 10000), 'k', alpha=0.1)
    ax_h0.plot(np.linspace(0, 1, 10000), np.linspace(0, 1, 10000), 'k', alpha=0.1)

    ax_h1.set_xlabel('p-value')
    ax_h0.set_xlabel('p-value')
    ax_h1.set_title('Simulated p-value CDFs under H1')
    ax_h0.set_title('Simulated p-value CDFs under H0')

    ax_h1.axvline(0.05, color='k', alpha=0.5)

    lines = []

    for title, (ab_pvals, aa_pvals, linetype) in dict2plot.items():
        line, = plot_cdf(ab_pvals, title, ax_h1, linetype)
        plot_cdf(aa_pvals, title, ax_h0, linetype)
        lines.append(line)
    ax_legend.legend(handles=lines, loc='center')
    ax_legend.axis('off')

    ax_powers.set_title('Test Power')
    tests_powers = []
    tests_labels = []
    tests_colours = []
    for title, (ab_pvals, _, linetype) in dict2plot.items():
        tests_labels.append(title)
        tests_colours.append(linetype[:1])
        tests_powers.append(np.mean(ab_pvals < 0.05))
    ax_powers.barh(np.array(tests_labels), np.array(tests_powers), color=np.array(tests_colours))

    ax_attempts.hist(attempts_0[:10].flatten(), 100, (0, 100), density=True)
    attempts_std = np.std(attempts_0[:10].flatten())
    ax_attempts.set_title('Attempts (views) distribution, std = {:<20.0f}'.format(attempts_std))

    if fix_axis:
        ax_successes.hist(ground_truth_success_rates[:10].flatten(), 100, (0, 0.5), density=True)
    else:
        ax_successes.hist(ground_truth_success_rates[:10].flatten(), bins=100, density=True)

    success_rate_std = ground_truth_success_rates[:10].flatten().std()
    ax_successes.set_title('user-CTR, std = {:2.3f}'.format(success_rate_std))
    return fig


def plot_from_params(data_dir: str, params: Dict, codenames: Optional[Set[str]] = None):
    gen_params = dict(params)
    gen_params['NN'] = 10
    (attempts_0, _), _, ground_truth_success_rates = generate_data(**gen_params)

    required_codenames2titles = {}
    if codenames is not None:
        for k, v in codenames2titles.items():
            if k in codenames:
                required_codenames2titles[k] = v
    else:
        required_codenames2titles.update(codenames2titles)

    dict2plot = {}
    for codename, (title, linetype) in required_codenames2titles.items():
        ab_data, aa_data = rpv(data_dir, codename, **params)
        dict2plot[title] = (ab_data, aa_data, linetype)

    fig = plot_summary(dict2plot, attempts_0, ground_truth_success_rates)
    return fig


def frame_from_params(data_dir: str, param: Dict, codenames: Optional[Set[str]] = None):
    fig = plot_from_params(data_dir, param, codenames)
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image


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

    (attempts_0_ab, successes_0_ab), (attempts_1_ab, successes_1_ab), gt_success_rates = generate_data(**ab_params)
    (attempts_0_aa, successes_0_aa), (attempts_1_aa, successes_1_aa), _ = generate_data(**aa_params)

    wpv(data_dir, 'ttest_successes_count', lambda: t_test(successes_0_ab, successes_1_ab),
        lambda: t_test(successes_0_aa, successes_1_aa),
        **ab_params)
    wpv(data_dir, 'mannwhitney_successes_count', lambda: mannwhitney(successes_0_ab, successes_1_ab),
        lambda: mannwhitney(successes_0_aa, successes_1_aa), **ab_params)

    wpv(data_dir, 'delta', lambda: delta_method_ctrs(successes_0_ab, attempts_0_ab, successes_1_ab, attempts_1_ab),
        lambda: delta_method_ctrs(successes_0_aa, attempts_0_aa, successes_1_aa, attempts_1_aa), **ab_params)

    wpv(data_dir, 'bootstrap',
        lambda: bootstrap(successes_0_ab / attempts_0_ab, attempts_0_ab, successes_1_ab / attempts_1_ab, attempts_1_ab),
        lambda: bootstrap(successes_0_aa / attempts_0_aa, attempts_0_aa, successes_1_aa / attempts_1_aa, attempts_1_aa),
        **ab_params)

    linearized_0_ab, linearized_1_ab = linearization_of_successes(successes_0_ab, attempts_0_ab, successes_1_ab,
                                                                  attempts_1_ab)
    linearized_0_aa, linearized_1_aa = linearization_of_successes(successes_0_aa, attempts_0_aa, successes_1_aa,
                                                                  attempts_1_aa)
    wpv(data_dir, 'linearization', lambda: t_test(linearized_0_ab, linearized_1_ab),
        lambda: t_test(linearized_0_aa, linearized_1_aa),
        **ab_params)

    wpv(data_dir, 'buckets', lambda: buckets(successes_0_ab / attempts_0_ab, np.ones(shape=attempts_0_ab.shape),
                                             successes_1_ab / attempts_1_ab, np.ones(shape=attempts_1_ab.shape)),
        lambda: buckets(successes_0_aa / attempts_0_aa, np.ones(shape=attempts_0_aa.shape),
                        successes_1_aa / attempts_1_aa, np.ones(shape=attempts_1_aa.shape)), **ab_params)

    wpv(data_dir, 't_test_ctrs', lambda: t_test(successes_0_ab / attempts_0_ab, successes_1_ab / attempts_1_ab),
        lambda: t_test(successes_0_aa / attempts_0_aa, successes_1_aa / attempts_1_aa), **ab_params)

    corr_aware_w_0_ab, corr_aware_w_1_ab = intra_user_correlation_aware_weights(successes_0_ab, attempts_0_ab,
                                                                                attempts_1_ab)
    corr_aware_w_0_aa, corr_aware_w_1_aa = intra_user_correlation_aware_weights(successes_0_aa, attempts_0_aa,
                                                                                attempts_1_aa)

    wpv(data_dir, 'weighted_bootstrap',
        lambda: bootstrap(successes_0_ab / attempts_0_ab, corr_aware_w_0_ab, successes_1_ab / attempts_1_ab,
                          corr_aware_w_1_ab),
        lambda: bootstrap(successes_0_aa / attempts_0_aa, corr_aware_w_0_aa, successes_1_aa / attempts_1_aa,
                          corr_aware_w_1_aa),
        **ab_params)

    wpv(data_dir, 'weighted_linearization',
        lambda: t_test(linearized_0_ab * corr_aware_w_0_ab, linearized_1_ab * corr_aware_w_1_ab),
        lambda: t_test(linearized_0_aa * corr_aware_w_0_aa, linearized_1_aa * corr_aware_w_1_aa), **ab_params)

    wpv(data_dir, 'weighted_t_test_ctrs',
        lambda: t_test(successes_0_ab / attempts_0_ab * corr_aware_w_0_ab,
                       successes_1_ab / attempts_1_ab * corr_aware_w_1_ab),
        lambda: t_test(successes_0_aa / attempts_0_aa * corr_aware_w_0_aa,
                       successes_1_aa / attempts_1_aa * corr_aware_w_1_aa),
        **ab_params)

    wpv(data_dir, 'weighted_buckets',
        lambda: buckets(successes_0_ab / attempts_0_ab, corr_aware_w_0_ab, successes_1_ab / attempts_1_ab,
                        corr_aware_w_1_ab),
        lambda: buckets(successes_0_aa / attempts_0_aa, corr_aware_w_0_aa, successes_1_aa / attempts_1_aa,
                        corr_aware_w_1_aa),
        **ab_params)

    wpv(data_dir, 'weighted_sqr_bootstrap',
        lambda: bootstrap(successes_0_ab / attempts_0_ab, np.sqrt(attempts_0_ab), successes_1_ab / attempts_1_ab,
                          np.sqrt(attempts_1_ab)),
        lambda: bootstrap(successes_0_aa / attempts_0_aa, np.sqrt(attempts_0_aa), successes_1_aa / attempts_1_aa,
                          np.sqrt(attempts_1_aa)),
        **ab_params)

    wpv(data_dir, 'weighted_sqr_linearization',
        lambda: t_test(linearized_0_ab * np.sqrt(attempts_0_ab), linearized_1_ab * np.sqrt(attempts_1_ab)),
        lambda: t_test(linearized_0_aa * np.sqrt(attempts_0_aa), linearized_1_aa * np.sqrt(attempts_1_aa)),
        **ab_params)

    wpv(data_dir, 'weighted_sqr_t_test_ctrs',
        lambda: t_test(successes_0_ab / attempts_0_ab * np.sqrt(attempts_0_ab),
                       successes_1_ab / attempts_1_ab * np.sqrt(attempts_1_ab)),
        lambda: t_test(successes_0_aa / attempts_0_aa * np.sqrt(attempts_0_aa),
                       successes_1_aa / attempts_1_aa * np.sqrt(attempts_1_aa)),
        **ab_params)

    wpv(data_dir, 'weighted_sqr_buckets',
        lambda: buckets(successes_0_ab / attempts_0_ab, np.sqrt(attempts_0_ab), successes_1_ab / attempts_1_ab,
                        np.sqrt(attempts_1_ab)),
        lambda: buckets(successes_0_aa / attempts_0_aa, np.sqrt(attempts_0_aa), successes_1_aa / attempts_1_aa,
                        np.sqrt(attempts_1_aa)),
        **ab_params)

    wpv(data_dir, 'weighted_sqr_buckets',
        lambda: buckets(successes_0_ab / attempts_0_ab, np.sqrt(attempts_0_ab), successes_1_ab / attempts_1_ab,
                        np.sqrt(attempts_1_ab)),
        lambda: buckets(successes_0_aa / attempts_0_aa, np.sqrt(attempts_0_aa), successes_1_aa / attempts_1_aa,
                        np.sqrt(attempts_1_aa)),
        **ab_params)

    smoothed_ctrs_0_ab, smoothed_ctrs_1_ab = get_smoothed_ctrs(successes_0_ab, attempts_0_ab,
                                                               successes_1_ab, attempts_1_ab)
    smoothed_ctrs_0_aa, smoothed_ctrs_1_aa = get_smoothed_ctrs(successes_0_aa, attempts_0_aa,
                                                               successes_1_aa, attempts_1_aa)
    wpv(data_dir, 'ttest_smoothed',
        lambda: t_test(smoothed_ctrs_0_ab, smoothed_ctrs_1_ab),
        lambda: t_test(smoothed_ctrs_0_aa, smoothed_ctrs_1_aa),
        **ab_params)
