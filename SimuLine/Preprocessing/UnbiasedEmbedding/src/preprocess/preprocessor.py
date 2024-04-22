"""
Codes for preprocessing datasets used in the real-world experiments
in the paper "Unbiased Pairwise Learning from Biased Implicit Feedback".
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split

RAW_DIR = './SimuLine/Preprocessing/UnbiasedEmbedding/data/raw'
POINT_DIR = './SimuLine/Preprocessing/UnbiasedEmbedding/data/point'
PAIR_DIR = './SimuLine/Preprocessing/UnbiasedEmbedding/data/pair'


def preprocess_dataset(data: str):
    """Load and preprocess datasets."""
    print('Loading ...')
    np.random.seed(12345)
    # train: u-i, ndarray
    with open(RAW_DIR + '/train.txt', 'r') as f:
        train_str = f.read()
        train = train_str.split('\n')
        train = [record.split('\t') for record in train]
        train = [[int(record[0]), int(record[1]), 1] for record in train]
        train_array = np.array(train)
    # test: u-i-r, ndarray
    with open(RAW_DIR + '/test.txt', 'r') as f:
        test_str = f.read()
        test = test_str.split('\n')
        test = [record.split('\t') for record in test]
        test = [[int(record[0]), int(record[1]), 1] for record in test]
        test_array = np.array(test)
    # num count
    num_users = max(train_array[:, 0].max(), test_array[:, 0].max()) + 1
    num_items = max(train_array[:, 1].max(), test_array[:, 1].max()) + 1
    # add random negative
    random_negative_train = np.random.randint(0, num_items, size=len(train_array))
    for i, record in enumerate(train_array):
        train.append([record[0], random_negative_train[i], 0])
    train = np.array(train)
    random_negative_test = np.random.randint(0, num_items, size=len(test_array))
    for i, record in enumerate(test_array):
        test.append([record[0], random_negative_test[i], 0])
    test = np.array(test)
    # estimate propensities and user-item frequencies.
    print('Estimate propensities and user-item frequencies.')
    user_freq = np.unique(np.r_[train, test][np.r_[train, test][:, 2] == 1, 0], return_counts=True)[1]
    item_freq = np.unique(np.r_[train, test][np.r_[train, test][:, 2] == 1, 1], return_counts=True)[1]
    pscore = (item_freq / item_freq.max()) ** 0.5
    # train-val split using the raw training datasets
    print('Train-Val split using the raw training datasets')
    train, val = train_test_split(train, test_size=0.1, random_state=12345)
    # save preprocessed datasets
    # pointwise
    np.save(file='{}/train.npy'.format(POINT_DIR), arr=train.astype(np.int))
    np.save(file='{}/val.npy'.format(POINT_DIR), arr=val.astype(np.int))
    np.save(file='{}/test.npy'.format(POINT_DIR), arr=test)
    np.save(file='{}/pscore.npy'.format(POINT_DIR), arr=pscore)
    np.save(file='{}/user_freq.npy'.format(POINT_DIR), arr=user_freq)
    np.save(file='{}/item_freq.npy'.format(POINT_DIR), arr=item_freq)
    # pairwise
    print('Build Pairwise Data')
    samples = 10
    bpr_train = _bpr(data=train, n_samples=samples)
    ubpr_train = _ubpr(data=train, pscore=pscore, n_samples=samples)
    bpr_val = _bpr(data=val, n_samples=samples)
    ubpr_val = _ubpr(data=val, pscore=pscore, n_samples=samples)
    pair_test = _bpr_test(data=test, n_samples=samples)
    np.save(file='{}/bpr_train.npy'.format(PAIR_DIR), arr=bpr_train)
    np.save(file='{}/ubpr_train.npy'.format(PAIR_DIR), arr=ubpr_train)
    np.save(file='{}/bpr_val.npy'.format(PAIR_DIR), arr=bpr_val)
    np.save(file='{}/ubpr_val.npy'.format(PAIR_DIR), arr=ubpr_val)
    np.save(file='{}/test.npy'.format(PAIR_DIR), arr=pair_test)


def _bpr(data: np.ndarray, n_samples: int) -> np.ndarray:
    """Generate training data for the naive bpr."""
    print('Generate training data for the naive bpr.')
    df = pd.DataFrame(data, columns=['user', 'item', 'click'])
    positive = df.query("click == 1")
    negative = df.query("click == 0")
    ret = positive.merge(negative, on="user")\
        .sample(frac=1, random_state=12345)\
        .groupby(["user", "item_x"])\
        .head(n_samples)

    return ret[['user', 'item_x', 'item_y']].values


def _bpr_test(data: np.ndarray, n_samples: int) -> np.ndarray:
    """Generate training data for the naive bpr."""
    print('Generate training data for the naive bpr.')
    df = pd.DataFrame(data, columns=['user', 'item', 'gamma'])
    ret = df.merge(df, on="user")\
        .sample(frac=1, random_state=12345)\
        .groupby(["user", "item_x"])\
        .head(n_samples)

    return ret[['user', 'item_x', 'item_y', 'gamma_x', 'gamma_y']].values


def _ubpr(data: np.ndarray, pscore: np.ndarray, n_samples: int) -> np.ndarray:
    """Generate training data for the unbiased bpr."""
    print('Generate training data for the unbiased bpr.')
    data = np.c_[data, pscore[data[:, 1].astype(int)]]
    df = pd.DataFrame(data, columns=['user', 'item', 'click', 'theta'])
    positive = df.query("click == 1")
    ret = positive.merge(df, on="user")\
        .sample(frac=1, random_state=12345)\
        .groupby(["user", "item_x"])\
        .head(n_samples)
    ret = ret[ret["item_x"] != ret["item_y"]]

    return ret[['user', 'item_x', 'item_y', 'click_y', 'theta_x', 'theta_y']].values
