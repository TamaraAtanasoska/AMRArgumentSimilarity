from collections import Counter

import numpy as np
from sklearn.metrics import f1_score
from random import shuffle


def find_threshold(candidates, X, Y):
    high_score = 0.0
    threshold = 0.0
    for t in candidates:
        x_copy = X.copy()
        x_copy[X < t] = 0
        x_copy[X >= t] = 1
        score = f1_score(x_copy, Y, average="macro", zero_division=0)
        if score > high_score:
            high_score = score
            threshold = t
    return threshold


def score_test(threshold, X, Y):
    x_copy = X.copy()
    x_copy[X < threshold] = 0
    x_copy[X >= threshold] = 1
    score_macro = f1_score(x_copy, Y, average="macro", zero_division=0)
    score_1 = f1_score(x_copy, Y, pos_label=1, zero_division=0)
    score_0 = f1_score(x_copy, Y, pos_label=0, zero_division=0)
    return score_macro, score_1, score_0


def runcv(ybar, y, topics, fold_size):
    # define possible thresholds 0.01, 0.02, ...
    trs = np.linspace(0, 1.0, 100)

    topicids = list(set(topics))
    n_topics = len(topicids)
    assert n_topics % fold_size == 0, f'n_topics {n_topics} must be divisible by fold_size {fold_size}'

    out1 = []
    out2 = []
    out3 = []
    thresholds = []

    # do 10 random runs
    for run in range(10):
        out1t = []
        out2t = []
        out3t = []

        # shuffle topics
        shuffle(topicids)

        # threshold cross validation over topics
        for i in range(0, n_topics, fold_size):
            # split train/test
            tet = topicids[i:i + fold_size]
            trt = topicids[0:i] + topicids[i + fold_size:]
            tet_ids = np.where(np.isin(topics, tet))
            trt_ids = np.where(np.isin(topics, trt))
            ybar_train = ybar[trt_ids]
            ybar_test = ybar[tet_ids]
            y_train = y[trt_ids]
            y_test = y[tet_ids]

            # to numpy array
            ybar_train = np.array(ybar_train)
            ybar_test = np.array(ybar_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)

            threshold = find_threshold(trs, ybar_train, y_train)
            thresholds.append(threshold)
            score_macro, score_1, score_0 = score_test(threshold, ybar_test, y_test)

            # collect result for test fold
            out1t.append(score_macro)
            out2t.append(score_1)
            out3t.append(score_0)

        # collect average over test folds
        out1.append(np.mean(out1t))
        out2.append(np.mean(out2t))
        out3.append(np.mean(out3t))

    # average over random runs, print
    print("final score", np.mean(out1), "&", np.mean(out2), "&", np.mean(out3))
    threshold_counts = Counter(thresholds)
    print('thresholds:', threshold_counts)
    final_threshold = sum([(k * v)/sum(threshold_counts.values()) for k, v in threshold_counts.items()])
    print('weighted threshold:', final_threshold)
    return np.mean(out1), np.mean(out2), np.mean(out3), final_threshold
