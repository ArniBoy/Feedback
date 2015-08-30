#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Arne Recknagel'

import logging

from preprocessing import POS, NEG, NEU
from backfeed import Feeder
from external_sources import AFinnWordList
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
    logging.info('f1 scores\n\tpositive: %f\n\tneutral: %f\n\tnegative: %f\n\tmacro: %f'
                 % (pos_f1, neg_f1, neu_f1, (pos_f1+neg_f1)/2.0))
    logging.info('accuracy scores\n\tpositive: %f\n\tneutral: %f\n\tnegative: %f\n\tmacro: %f'
                 % (pos_acc, neg_acc, neu_acc, (pos_acc+neg_acc)/2.0))


def run(model, x_train, y_train, x_test, y_test):
    # initial step
    model.fit(x_train, y_train)
    logging.info('initial evaluation')
    get_score(model, x_test, y_test)

    # retrain routine set-up
    feed = Feeder()
    af_wl = AFinnWordList(root+'Data/afinn/AFINN-111.txt')
    af_wl.add_filter_ranges(**{str(POS): (1, float('inf')),
                               str(NEG): (float('-inf'), -1),
                               str(NEU): (-1, 1)})
    af_wl.add_weight(1, model.best_estimator_.named_steps['svm'].classes_)
    feed.add_mutator(af_wl)

    # retrain loop
    for i in range(30):
        logging.debug('count nr. %i' % i)

        # feedback
        feed.add_best_n(model, 300, x_train, y_train, False, ['weight'])

        # evaluation
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
    run(model, train[0], train[1], dev[0], dev[1])


if __name__ == '__main__':
    main()
