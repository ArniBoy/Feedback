#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Arne Recknagel'

import logging
from math import sqrt
import codecs

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN
from sklearn import metrics

from feature_extractors import DataSeparator
from preprocessing import parse, POS, NEG, NEU
from util import svm_pipeline, k_means_pipeline, get_corpus, init_logging, root, get_feature_union, get_corpus
init_logging()


def get_train_data():
    classes = POS | NEU | NEG
    train_loc = root+'Data/twitterData/train_alternative.tsv'
    dev_loc = root+'Data/twitterData/dev_alternative.tsv'
    test_loc = root+'Data/twitterData/test_alternative.tsv'
    train_labels, train_tweets, train_pos = parse(
        train_loc, classes
    )
    dev_labels, dev_tweets, dev_pos = parse(
        dev_loc, classes
    )
    test_labels, test_tweets, test_pos = parse(
        test_loc, classes
    )
    semeval_train = [e[0]+'\t'+e[1] for e in zip(train_tweets, train_pos)], train_labels
    semeval_dev = [e[0]+'\t'+e[1] for e in zip(dev_tweets, dev_pos)], dev_labels
    semeval_test = [e[0]+'\t'+e[1] for e in zip(test_tweets, test_pos)], test_labels
    return semeval_train, semeval_dev, semeval_test


def draw(data, latex=True):
    logging.info('starting distance measurement')
    trans = Pipeline([
        ('separate', DataSeparator()),
        ('features', get_feature_union()),
    ])
    feature_space = trans.fit_transform(data)
    distances = metrics.pairwise.pairwise_distances(feature_space, metric='cosine')
    logging.info('distances computed')
    buckets = {}
    for idx_a, row in enumerate(distances):
        for idx_b, val in enumerate(row):
            if idx_a != idx_b:
                rounded = float('%.2f' % val)
                if rounded not in buckets:
                    buckets[rounded] = 1
                else:
                    buckets[rounded] += 1
    vars = 0
    num_instances = 0
    for key, val in buckets.items():
        num_instances += val
        vars += key * val
    logging.info('average variance: %.3f' % (vars/num_instances))
    max_length = 160.0
    max_count = max(buckets.values())
    if not latex:
        for key in sorted(buckets.keys()):
            num_bar = int(buckets[key] * max_length / max_count)
            logging.info('%s: %s' % (key, num_bar * '-'))
    else:
        to_print = ''
        for count, key in enumerate(sorted(buckets.keys())):
            num_bar = int(buckets[key] * max_length / max_count)
            to_print += '(%s,%s)' % (key, num_bar)
            if count % 10 == 0:
                to_print += '\n'
        logging.info(to_print)

def dbscan(feature_space):
    minimum_per_cluster = 1.0 / 4
    samples_per_cluster = int(feature_space.shape[0] / sqrt(feature_space.shape[0]/2.0) * minimum_per_cluster)
    logging.info('Estimated minimum amount of samples per cluster: %i' % samples_per_cluster)
    eps = (0.3, 0.6, 0.9, 1.10, 1.42, 1.74, 2.01, 2.24, 2.45, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 3.75,
           4.00, 4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 6.75, 7.00, 7.25, 7.50, 7.75, 8.00)
    for elem in eps:
        db = DBSCAN(eps=elem, min_samples=minimum_per_cluster, metric='cosine', algorithm='brute').fit(feature_space)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        logging.info('Estimated number of clusters for esp=%.3f: %d' % (elem, n_clusters_))

        for label in set(labels):
            labelled_samples = (labels == label) & core_samples_mask
            for idx in [i for i in labelled_samples if i is not False]:
                pass
                # TODO find central cluster samples?


def label_counter(cluster_collection, model):
    total_length = float(sum([len(data) for data in zip(*cluster_collection)[1]]))
    consistency = 0
    for c, data in cluster_collection:
        logging.debug('cluster: %s content length: %s' % (c, len(data)))
        y_pred = model.predict(data)
        label_counts = {}
        for p in y_pred:
            if p in label_counts:
                label_counts[p] += 1
            else:
                label_counts[p] = 1
        consistency += max(label_counts.values()) / total_length
    return consistency


def k_means():
    k = 8
    data = get_corpus(20000)
    baseline_clf = svm_pipeline()
    train, dev, test = get_train_data()
    baseline_clf.fit(train[0], train[1])

    max_consistency = label_counter([(0, data)], baseline_clf)
    logging.info('baseline consistency: %f' % max_consistency)

    tmp = [
        (tweet.strip(), int(label.strip())) for tweet, label in zip(
            codecs.open(root+'Data/Corpora/batches/tokenized.tsv', 'r,', 'utf-8'),
            codecs.open(root+'Data/logs/cluster.txt', 'r,', 'utf-8'))
    ]
    clf = {}
    for tweet, label in tmp:
        if label not in clf:
            clf[label] = [tweet]
        else:
            clf[label].append(tweet)
    max_consistency = label_counter(clf.items(), baseline_clf)
    logging.info('max consistency: %f' % max_consistency)

    for label in clf.keys():
        logging.info(label)
        frq = {}
        for tw in clf[label]:
            try:
                est = baseline_clf.predict([tw])[0]
                if est in frq:
                    frq[est] += 1
                else:
                    frq[est] = 1
            except ValueError:
                logging.warn('could not parse tweet %s' % tw)
        logging.info(frq)

    max_clusters = ()
    km = k_means_pipeline(k)
    for run in range(50):
        logging.debug('run number %i' % run)
        km.fit(data)

        clusters = {idx: [] for idx in range(k)}
        for entry, cluster in zip(data, km.named_steps['k_means'].labels_):
            clusters[cluster].append(entry)
        local_consistency = label_counter(clusters.items(), baseline_clf)

        if local_consistency > max_consistency:
            max_consistency = local_consistency
            max_clusters = km.named_steps['k_means'].labels_
            logging.debug('max consistency updated to %f' % max_consistency)

    logging.info('most consistent score: %f' % max_consistency)
    with open(root+'Data/logs/cluster.txt', 'w') as log:
        for line in max_clusters:
            log.write('%s\n' % line)


if __name__ == '__main__':
    pass

