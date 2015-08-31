#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Arne Recknagel'

import logging
import pickle

from preprocessing import POS, NEG, NEU
from backfeed import Feeder
from external_sources import AFinnWordList, SerelexCluster, AutoCluster
from util import svm_pipeline, get_final_semeval_data, f1, init_logging, root
init_logging()


def get_score(model, x_test, y_test):
    """
    Computes the f1 and accuracy score for a model trained on three classes and weighting on two
    :param model: trained classifier
    :param x_test: test data
    :param y_test: test labels
    """
    predicted = model.predict(x_test)
    pos_f1, pos_acc = f1(predicted, y_test, POS)
    neg_f1, neg_acc = f1(predicted, y_test, NEU)
    neu_f1, neu_acc = f1(predicted, y_test, NEG)

    logging.info('f1 scores\n\tpositive: %f\n\tnegative: %f\n\tneutral: %f\n\tmacro: %f'
                 % (pos_f1, neg_f1, neu_f1, (pos_f1+neg_f1)/2.0))
    logging.info('accuracy scores\n\tpositive: %f\n\tnegative: %f\n\tneutral: %f\n\tmacro: %f'
                 % (pos_acc, neg_acc, neu_acc, (pos_acc+neg_acc)/2.0))


def run(model, x_train, y_train, x_test, y_test, mode, retrain=30, amount=300, token=''):
    # initial step
    model.fit(x_train, y_train)
    logging.info('initial evaluation')
    get_score(model, x_test, y_test)

    # external sources set-up
    cl = pickle.load(open('cl.model', 'rb'))
    af = pickle.load(open('af.model', 'rb'))
    km = pickle.load(open('km.model', 'rb'))
    classes = model.best_estimator_.named_steps['svm'].classes_
    cl.add_filter_ranges(**{str(POS): (1.5, float('inf')),
                            str(NEG): (float('-inf'), -1.5),
                            str(NEU): (-1.5, 1.5)})
    cl.add_weight(5, classes)
    af.add_filter_ranges(**{str(POS): (0.4, float('inf')),
                            str(NEG): (float('-inf'), -0.4),
                            str(NEU): (-0.4, 0.4)})
    af.add_weight(2.2, classes)
    km.add_filter_ranges(**{str(POS): (2.5, float('inf')),
                            str(NEG): (float('-inf'), -2.5),
                            str(NEU): (-2.5, 2.5)})
    km.add_weight(40, classes)

    # source inclusion
    feed = Feeder()
    if token == 'km':
        feed.add_mutator(km)
    if token == 'af':
        feed.add_mutator(af)
    if token == 'cl':
        feed.add_mutator(cl)

    # retrain loop, feedback, and evaluation
    for i in range(retrain):
        logging.debug('count nr. %i' % i)
        feed.add_best_n(model, amount, x_train, y_train, False, mode)
        get_score(model, x_test, y_test)


def main():
    # load labelled data
    classes = POS | NEU | NEG
    train_loc = root+'Data/twitterData/train_alternative.tsv'
    dev_loc = root+'Data/twitterData/dev_alternative.tsv'
    test_loc = root+'Data/twitterData/test_alternative.tsv'
    train, dev, test = get_final_semeval_data(classes, train_loc, dev_loc, test_loc)

    # load model
    model = svm_pipeline()

    # run main routine
    for funrun in ('km', 'af', 'cl'):
        run(model, train[0], train[1], dev[0], dev[1], mode=['filter'], retrain=5,
            token=funrun)


if __name__ == '__main__':
    main()
