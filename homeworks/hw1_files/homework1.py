import scipy.io
import time
from myRecommender import my_recommender
import numpy as np


def rmse(u, v, mat):
    mask = mat > 0
    res = np.sum(((u.dot(v.T) - mat) * mask) ** 2) / float(np.sum(mask))
    return np.sqrt(res)


cell = scipy.io.loadmat('movie_data.mat')
rate_mat = cell['train']
test_mat = cell['test']

low_rank_ls = [1, 3, 5]
for lr in low_rank_ls:
    for reg_flag in [False, True]:
        st = time.time()
        U, V = my_recommender(rate_mat, lr, reg_flag)

        t = time.time() - st

        print('SVD-%s-%i\t%.4f\t%.4f\t%.2f\n' % ('withReg' if reg_flag else 'noReg', lr,
                                                 rmse(U, V, rate_mat), rmse(U, V, test_mat), t))

