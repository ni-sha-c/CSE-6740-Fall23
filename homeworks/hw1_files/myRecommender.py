import numpy as np


def my_recommender(rate_mat, lr, with_reg):
    """

    :param rate_mat:
    :param lr:
    :param with_reg:
        boolean flag, set true for using regularization and false otherwise
    :return:
    """

    # TODO pick hyperparams
    max_iter = 0
    learning_rate = 0
    reg_coef = 0
    n_user, n_item = rate_mat.shape[0], rate_mat.shape[1]

    U = np.random.rand(n_user, lr) / lr
    V = np.random.rand(n_item, lr) / lr

    # TODO implement your code here

    return U, V