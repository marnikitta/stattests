import numpy as np

from stattests.utils import apply_all_tests

if __name__ == '__main__':
    success_rate = 0.02
    uplift = 0.1
    beta = 10
    N = 5000
    NN = 2000
    skew_start, skew_end = 1, 10
    points = 20

    for skew in np.linspace(skew_start, skew_end, points):
        apply_all_tests('./data', NN, N, uplift, success_rate, beta, skew)
