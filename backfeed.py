#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Arne Recknagel'

import logging
from codecs import open
from random import shuffle
import numpy as np

from preprocessing import get_tweet
from external_sources import SentMutate
from toy_classifier import root


class Feeder:
    """
    Class which encapsulates the feedback module. Currently, only a 50.000 sized
    subset is loaded.
    """
    def __init__(self, corpus_loc, limit=20000):
        """
        Constructor for a Feeder instance
        :param corpus_loc: Location of the sent140 corpus
        :param limit: Cutoff for testing, lower value improves speed
        """
        self.current_idx = 0
        self.corpus = {}
        count = 0
        try:
            for count, e in enumerate(open(corpus_loc, 'r', 'utf-8')):
                count += 1
                if count > limit:  # restrict input to a few tweets, not all yet
                    break
                self.corpus[count] = '\t'.join(get_tweet(e)[1:])
        except UnicodeDecodeError:
            logging.warning('UnicodeDecodeError in %s at line %i' %
                            (corpus_loc, count))
        self.mutators = []

    def add_mutator(self, mut):
        """
        Build a mutation scheme by adding instances of SentMutate objects
        :param mut: Mutator to be added
        """
        assert isinstance(mut, SentMutate)
        self.mutators.append(mut)

    def valid_distances(self, model):
        """
        Wraps sklearn's decision_function so that binary classification also
        returns a list of shape (n_samples, n_classes), instead of a
        single-entry list.
        :param model: Used classifier
        :return: Probabilities as returned by the classifier's decision_function
        """
        distances = model.decision_function(self.corpus.values())
        if isinstance(distances[0], np.float64):
            tmp = np.zeros((len(distances), 2))
            tmp[:, 0] = -distances
            tmp[:, 1] = distances
            distances = tmp
        return distances

    def diff(self, seq):
        """
        This function is used to sort a sequence of tuples in such a way, that
        those sequence entries where the tuple entry at position
        self.current_idx has the highest value in comparison to all other tuple
        entries are on top. Pass this as the sort key to a list sort.
        :param seq: the tuples, we don't see the full list from in here
        :return: A high value if seq[1][self.current_idx] 'wins'
        """
        avg_diff = 0.0
        for val in seq[1]:
            avg_diff += seq[1][self.current_idx] - val
        return avg_diff / (len(seq[1])-1)

    def add_best_n(self, model, num, x, y, append, modes):
        """
        Finds the top 'num' entries for model evaluation given a positive and a
        negative class. After adding them to the existing training data, they
        are removed from the corpus.
        :param model: Trained model
        :param num: Number of total new data points which get added
        :param x: Training data
        :param y: Training labels
        """
        # get class distances for the corpus (might take a while..)
        distances = zip(
            self.corpus.keys(), self.valid_distances(model)
        )

        # optional distance re-weighting
        if 'weight' in modes:
            for mut in self.mutators:
                for idx in range(len(distances)):
                    key, dist = distances[idx]
                    tweet = self.corpus[key].split('\t')[0]
                    mut.apply_weighting(tweet, dist)

        # label entries
        class_labels = model.best_estimator_.named_steps['svm'].classes_
        amount = int(num/len(class_labels))
        to_add = []
        for i, label in enumerate(class_labels):

            # set idx in sorting function to get current label
            self.current_idx = i
            tmp = [e[0] for e in sorted(distances, key=self.diff)]

            # potential hard filter
            if 'filter' in modes:
                for mut in self.mutators:
                    tmp = [key for key in tmp
                           if mut.apply_filter(self.corpus[key], label)]
            to_add += zip(tmp[-amount:], [label]*amount)

        # to_add gathered 'amount' entries per label
        shuffle(to_add)
        data, labels = map(
            list, zip(*[(self.corpus[key], l) for key, l in to_add])
        )

        if append:
            # add 'data' to x and y, removal from corpus probably irrelevant
            x += data
            y += labels
            for idx, _ in to_add:
                if idx in self.corpus:
                    del self.corpus[idx]
            model.fit(x, y)
        else:
            model.fit(x+data, y+labels)
