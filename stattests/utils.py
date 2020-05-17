from typing import Tuple, Dict, Optional, Set, List

import imageio
import numpy as np
import scipy.stats
import seaborn as sns
from IPython.core.display import HTML
from matplotlib import pyplot as plt

from stattests.data import rpv
from stattests.generation import generate_data

colors = sns.color_palette("bright")

codenames2titles = {
    'ttest_successes_count': ('T-test, clicks', colors[0], 'solid'),
    'mannwhitney_successes_count': ('Mann-Whitney test, clicks', colors[1], 'solid'),
    'delta': ('Delta-method, global CTR', colors[2], 'dashed'),
    'bootstrap': ('Bootstrap, global CTR', colors[3], 'dashed'),
    'linearization': ('Linearization, clicks', colors[4], 'solid'),
    'buckets_ctrs': ('Bucketization, global CTR', colors[5], 'dashed'),
    't_test_ctrs': ('T-test, user CTR', colors[6], 'dashed'),
    'weighted_bootstrap': ('Weighted bootstrap, global CTR', colors[7], 'solid'),
    'weighted_linearization': ('Weighted linearization, global CTR', colors[4], 'dotted'),
    'weighted_buckets': ('Weighted bucketization, global CTR', colors[5], 'dotted'),
    'weighted_t_test_ctrs': ('Weighted t-test, user CTR', colors[6], 'dotted'),
    'weighted_sqr_bootstrap': ('Weighted sqr bootstrap, global CTR', colors[3], 'dashdot'),
    'weighted_sqr_linearization': ('Weighted sqr linearization, global CTR', colors[4], 'dashdot'),
    'weighted_sqr_buckets': ('Weighted sqr bucketization, global CTR', colors[5], 'dashdot'),
    'weighted_sqr_t_test_ctrs': ('Weighted sqr t-test, user CTR', colors[6], 'dashdot'),
    'ttest_smoothed': ('T-test smoothed user CTR', colors[0], 'dashdot')
}

cdf_h1_title = 'Simulated p-value CDFs under H1 (Sensitivity)'
cdf_h0_title = 'Simulated p-value CDFs under H0 (FPR)'


def save_gif_and_show(path: str, frames: List[np.ndarray]):
    reversed_frames = frames.copy()
    reversed_frames.reverse()

    bounce_frames = frames + reversed_frames
    imageio.mimsave(path, bounce_frames, fps=2, format='GIF-PIL')
    return HTML(f'<img src="{path}" width="1000px">')


def plot_cdf(data, label, ax, color=colors[0], linestyle='solid'):
    sorted_data = np.sort(data)
    position = scipy.stats.rankdata(sorted_data, method='ordinal')
    cdf = position / data.shape[0]

    sorted_data = np.hstack((sorted_data, 1))
    cdf = np.hstack((cdf, 1))

    return ax.plot(sorted_data, cdf, color=color, linestyle=linestyle, label=label, linewidth=1.5)


def plot_summary(dict2plot: Dict[str, Tuple[np.ndarray, np.ndarray, str, str]],
                 attempts_0: np.ndarray,
                 ground_truth_success_rates: np.ndarray,
                 plot_params: Optional[Dict] = None):
    default_params = {'fix_axis': True, 'dpi': 100}
    plot_params = plot_params or {}

    for k, v in default_params.items():
        if k not in plot_params:
            plot_params[k] = v

    fig, ((ax_h1, ax_legend, ax_h0), (ax_powers, ax_attempts, ax_successes)) = plt.subplots(2, 3, figsize=(20, 10),
                                                                                            dpi=plot_params['dpi'])

    fig.subplots_adjust(wspace=0.2, hspace=0.3)

    ax_h1.plot(np.linspace(0, 1, 10000), np.linspace(0, 1, 10000), 'k', alpha=0.1)
    ax_h0.plot(np.linspace(0, 1, 10000), np.linspace(0, 1, 10000), 'k', alpha=0.1)

    ax_h1.set_xlabel('p-value')
    ax_h0.set_xlabel('p-value')
    ax_h1.set_title(cdf_h1_title)
    ax_h0.set_title(cdf_h0_title)

    ax_h1.set_ylabel('Sensitivity')
    ax_h0.set_ylabel('FPR')

    ax_h1.axvline(0.05, color='k', alpha=0.5)

    lines = []

    for title, (ab_pvals, aa_pvals, color, linestyle) in dict2plot.items():
        line, = plot_cdf(ab_pvals, title, ax_h1, color, linestyle)
        plot_cdf(aa_pvals, title, ax_h0, color, linestyle)
        lines.append(line)
    ax_legend.legend(handles=lines, loc='center')
    ax_legend.axis('off')

    ax_powers.set_title('Test Power')
    tests_powers = []
    tests_labels = []
    tests_colours = []
    for title, (ab_pvals, _, color, linestyle) in dict2plot.items():
        tests_labels.append(title)
        tests_colours.append(color)
        tests_powers.append(np.mean(ab_pvals < 0.05))
    ax_powers.barh(np.array(tests_labels), np.array(tests_powers), color=np.array(tests_colours))

    # ax_attempts.hist(attempts_0[:10].flatten(), 100, (0, 100), density=True)
    sns.distplot(attempts_0.flatten(), bins=range(0, 40), ax=ax_attempts,
                 kde=False, norm_hist=True)
    ax_attempts.set_xlim((0, 40))
    attempts_std = np.std(attempts_0[:10].flatten())
    ax_attempts.set_title('Views distribution, std = {:<20.0f}'.format(attempts_std))

    # ax_successes.hist(ground_truth_success_rates[:10].flatten(), 100, (0, 0.5), density=True)
    sns.distplot(ground_truth_success_rates.ravel(), bins=np.linspace(0, 0.2, 100), ax=ax_successes, kde=False,
                 norm_hist=True)
    ax_successes.set_xlim((0, 0.2))

    success_rate_std = ground_truth_success_rates[:10].flatten().std()
    ax_successes.set_title('Ground truth user CTR distribution, std = {:2.3f}'.format(success_rate_std))
    return fig


def plot_from_params(data_dir: str,
                     params: Dict,
                     codenames: Optional[Set[str]] = None,
                     plot_params: Optional[Dict] = None):
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
    for codename, (title, color, linestyle) in required_codenames2titles.items():
        ab_data, aa_data = rpv(data_dir, codename, **params)
        dict2plot[title] = (ab_data, aa_data, color, linestyle)

    fig = plot_summary(dict2plot, attempts_0, ground_truth_success_rates, plot_params)
    return fig


def frame_from_params(data_dir: str,
                      param: Dict,
                      codenames: Optional[Set[str]] = None,
                      plot_params: Optional[Dict] = None):
    fig = plot_from_params(data_dir, param, codenames, plot_params)
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image
