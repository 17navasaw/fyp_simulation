from __future__ import division
from sklearn.model_selection import train_test_split
import math
import numpy as np
from sklearn.metrics import classification_report
"""
====================================================================
Normal and Shrinkage Linear Discriminant Analysis for classification
====================================================================

Shows how shrinkage improves classification.
"""

import numpy as np
# import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
# from sklearn.lda import LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sys

n_train = 20  # samples for training
n_test = 200  # samples for testing
n_averages = 50  # how often to repeat classification
n_features_max = 75  # maximum number of features
step = 4  # step size for the calculation


def generate_data(n_samples, n_features):
    """Generate random blob-ish data with noisy features.

    This returns an array of input data with shape `(n_samples, n_features)`
    and an array of `n_samples` target labels.

    Only one feature contains discriminative information, the other features
    contain only noise.
    """
    X, y = make_blobs(n_samples=n_samples, n_features=1, centers=[[-2], [2]])

    # add non-discriminative features
    if n_features > 1:
        X = np.hstack([X, np.random.randn(n_samples, n_features - 1)])
    print(type(X))
    print(type(X[0]))
    print(X.shape)
    print(len(X))
    print(len(X[0]))
    print(len(y))
    print(y.shape)
    print(X)
    print(y)
    return X, y

# acc_clf1, acc_clf2 = [], []
# n_features_range = range(3, n_features_max + 1, step)
# for n_features in n_features_range:
#     score_clf1, score_clf2 = 0, 0
#     for _ in range(n_averages):
#         print("n_features: {0:5d}".format(n_features))
#         X, y = generate_data(n_train, n_features)
#
#         clf1 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X, y)
#         clf2 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None).fit(X, y)
#         sys.exit(0)
#         X, y = generate_data(n_test, n_features)
#         score_clf1 += clf1.score(X, y)
#         score_clf2 += clf2.score(X, y)
#
#     acc_clf1.append(score_clf1 / n_averages)
#     acc_clf2.append(score_clf2 / n_averages)

def extract_rms(window, index):
    sum_of_squares = 0
    for i in range(len(window)):
        sum_of_squares += math.pow(window[i][index], 2.0)

    return math.sqrt(sum_of_squares)


def extract_std(window, index):
    li = []
    for i in range(len(window)):
        li.append(window[i][index])

    return np.std(li, ddof=1)

def test():
    li = []
    li2 = [2, 4, 5]
    li3 = [6, 7, 8]
    li4 = [9, 10, 11]
    li5 = []
    li6 = [10, 11, 15]
    li7 = [20, 21, 22]
    li8 = [23, 24, 25]
    li9 = [26, 27, 28]
    li10 = [29, 30, 31]
    li.append(li2)
    li.append(li3)
    li.append(li4)
    li5.append(li6)
    li5.append(li7)
    li5.append(li8)
    li5.append(li9)
    li5.append(li10)
    # for i in range(0, 5):
    #     print(i)
    # print(np.fft.fft([2, 6, 9]))
    # print(list(map(lambda x: pow(x, 2), [2.0, 2.1, 2.2])))
    # print(list(map(abs, np.fft.fft([2, 6, 9]))))
    # print(np.fft.fftn(np.array(li), axes=[0, 1, 2]))
    # print(extract_std(li, 0))
    # li.extend(li5)
    # print(li)
    # li_train, li_test = train_test_split(li, test_size=1.0 / 3.0)
    li_train, li_test, li5_train, li5_test = train_test_split(li, li5, test_size=0.3, stratify=li5)
    print(li_train)
    print(li_test)
    print(li5_train)
    print(li5_test)


def testing(li_test):
    li2 = [2, 4, 5]
    li3 = [6, 7, 8]
    li_test.append(li2)
    li_test.append(li3)


def extract_rms(window, index):
    sum_of_squares = 0
    for i in range(len(window)):
        sum_of_squares += math.pow(window[i][index], 2.0)

    return math.sqrt(sum_of_squares)

y_true = [1, 2, 2, 2, 0]
y_pred = [2, 2, 1, 0, 0]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))