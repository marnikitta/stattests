import numpy as np
from matplotlib import pyplot as plt
import scipy.stats
    

def plot_cdf(data, label, ax, linetype):
    sorted_data = np.sort(data)
    position = scipy.stats.rankdata(sorted_data, method='ordinal')
    cdf = position / data.shape[0]

    sorted_data = np.hstack((sorted_data, 1))
    cdf = np.hstack((cdf, 1))

    ax.plot(sorted_data, cdf, linetype, label=label, linewidth=1.5)


def plot_summary(dict_to_plot_ab, dict_to_plot_aa, attempts_0, ground_truth_success_rates):
    """
    
    :param dict_to_plot_ab: dict[str, (np.array, str)]
    :param dict_to_plot_aa: dict[str, (np.array, str)]
    :param ground_truth_success_rates: np.array
    :param attempts_0: np.array
    :return: 
    """
    fig, (ax, ax_0, ax_1, ax_2, ax_3) = plt.subplots(5, 1, figsize=(16, 23))
    tests_powers = []
    tests_labels = []
    tests_colours = []
    for n, v in dict_to_plot_ab.items():
        p_vals, linetype = v
        plot_cdf(p_vals, n, ax, linetype)
        tests_powers.append(np.mean(p_vals < 0.05))
        tests_labels.append(n)
        tests_colours.append(linetype[:1])
    ax.plot(np.linspace(0, 1, 10000), np.linspace(0, 1, 10000), 'k', alpha=0.1)
    ax.set_xlabel('p-value')
    ax.axvline(0.05, color='k', alpha=0.5)
    ax.set_title('Simulated p-value CDFs under H1')
    ax.legend(loc='upper right')
    ind = list(range(len(tests_powers)))#np.argsort(tests_powers)
    ax_0.set_title('Test Power')
    ax_0.barh(np.array(tests_labels)[ind], np.array(tests_powers)[ind], color=np.array(tests_colours)[ind])
    for n, v in dict_to_plot_aa.items():
        p_vals, linetype = v
        plot_cdf(p_vals, n, ax_1, linetype)
    ax_1.plot(np.linspace(0, 1, 10000), np.linspace(0, 1, 10000), 'k', alpha=0.1)
    ax_1.set_xlabel('p-value')
    ax_1.set_title('Simulated p-value CDFs under H0')
    ax_1.legend(loc='upper right')
    # take only one experiment for plotting
    ax_2.hist(attempts_0[:10].flatten(), 100, (0, 100), density=True)
    attempts_std = attempts_0[:10].flatten().std()
    ax_2.set_title('Attempts (views) distribution, std = {:.3f}'.format(attempts_std))
    ax_3.hist(ground_truth_success_rates[:10].flatten(), bins=100)
    success_rate_std = ground_truth_success_rates[:10].flatten().std()
    ax_3.set_title('user-CTR, std = {:.3f}'.format(success_rate_std))
    return fig
    
