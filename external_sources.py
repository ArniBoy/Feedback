#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Arne Recknagel'

import logging
from codecs import open
from os import listdir, path

from preprocessing import POS, NEG, NEU
from util import k_means_pipeline, bucket_dist

root = '/Users/Ceca/Arne/Data'


class SentMutate(object):
    def __init__(self):
        """
        Sentiment mutator base class
        """
        self.ranges = {}
        self.weight = -1
        self.pos_idx = -1
        self.neg_idx = -1

    def add_filter_ranges(self, **kwargs):
        """
        Must be set if apply_filter is meant to be run.
        This one is a bit complicated. The keywords are label-values mapped to
        real number ranges, expressed as tuples, from small to big. It is used
        to assign the checked tweets to a label in the filter functions.
        e.g.:
        label pos: (2, 50),
        label neu: (-5, 5),
        label neg: (-50, -5)
        :param kwargs: label-range keyword arguments
        """
        self.ranges = {}
        for k in kwargs:
            self.ranges[str(k)] = kwargs[k]

    def add_weight(self, weight, classes):
        """
        Adds the weight this mutator has one the decision function values.
        :param weight: float, 1.0 should be normal
        :param classes: the classes_ attribute from the evaluating classifier
        """
        self.weight = weight
        for idx, label in enumerate(classes):
            if label == POS:
                self.pos_idx = idx
            if label == NEG:
                self.neg_idx = idx

    def apply_filter(self, tweet, label):
        """
        Interface for tweet filtering by sentiment weights
        :param tweet: input tweet, is converted to lowercase for filter
        application
        :param label: assumed label of the tweet
        :return: True if the tweet passed the filter, False if not
        """
        raise NotImplementedError("Class %s doesn't implement " %
                                  (self.__class__.__name__,))

    def apply_weighting(self, tweet, distances):
        """
        Interface for tweet weighting by sentiment weights -- operates directly
        on the distance list, so no return value is given
        :param tweet: input tweet, is converted to lowercase for weighting
        application
        :param distances: original distance measures for the tweet
        """
        raise NotImplementedError("Class %s doesn't implement " %
                                  (self.__class__.__name__,))


class AFinnWordList(SentMutate):
    def __init__(self, loc):
        """
        A mutator which employs the AFINN weighted word list
        (http://arxiv.org/abs/1103.2903)
        :param loc: Location of the weighted word list
        """
        super(AFinnWordList, self).__init__()
        self.w_l = {}
        for line in open(loc, 'r', 'utf-8'):
            k, v = line.strip().lower().split('\t')
            self.w_l[k] = int(v)

    def get_score(self, tweet):
        score = 0.0
        for word in tweet.lower().split(' '):
            sent = self.w_l.get(word, None)
            if sent is not None:
                score += sent
        return score

    def apply_filter(self, tweet, label):
        if not self.ranges:
            raise RuntimeError("Tried to run filter without specified ranges!")
        score = self.get_score(tweet)
        return self.ranges[str(label)][0] < score < self.ranges[str(label)][1]

    def apply_weighting(self, tweet, distances):
        if self.weight < 0 or self.pos_idx < 0 or self.neg_idx < 0:
            raise RuntimeError("Tried to run weighting without specified data!")
        score = self.get_score(tweet)
        word_count = len(tweet.split(' '))
        if score > 0:
            distances[self.pos_idx] += self.weight * score / word_count
        elif score < 0:
            distances[self.neg_idx] -= self.weight * score / word_count


class AutoCluster(SentMutate):
    def __init__(self, clusters_loc, sentiments_loc):
        """
        A mutator using automatically created word clusters as basis for
        re-weighting. Filter parameters are not needed, as class membership
        is binary.
        :param clusters_loc: location of the tweet-cluster association data
        :param sentiments_loc: location of the clusters-label association data
        """
        super(AutoCluster, self).__init__()
        self.clusters = k_means_pipeline(8)
        tmp = [
            (tweet.strip(), int(label.strip())) for tweet, label in zip(
                open(root+'/Corpora/batches/tokenized.tsv', 'r,', 'utf-8'),
                open(clusters_loc, 'r,', 'utf-8'))
        ]
        clf = {}
        for tweet, label in tmp:
            if label not in clf:
                clf[label] = [tweet]
            else:
                clf[label].append(tweet)
        for label in clf.keys():
            for tweet in clf[label]:
                print('%s: %s' % (label, tweet))
        exit()
        self.clusters.fit(*map(list, zip(*tmp)))
        self.cluster_sentiment = {idx: label.strip() for idx, label in open(sentiments_loc, 'r,', 'utf-8')}

    def get_score(self, tweet):
        cluster_sent = self.cluster_sentiment[self.clusters.predict(tweet)]
        if cluster_sent == 'positive':
            return POS
        elif cluster_sent == 'negative':
            return NEG
        elif cluster_sent == 'neutral':
            return NEU
        else:
            logging.warn('Input %s not format conform!' % cluster_sent)

    def apply_filter(self, tweet, label):
        return self.get_score(tweet) == label

    def apply_weighting(self, tweet, distances):
        if self.weight < 0 or self.pos_idx < 0 or self.neg_idx < 0:
            raise RuntimeError("Tried to run weighting without specified data!")
        verdict = self.get_score(tweet)
        if verdict == POS:
            distances[self.pos_idx] += self.weight
        elif verdict == NEG:
            distances[self.neg_idx] += self.weight


class SerelexCluster(SentMutate):
    def __init__(self, clusters_loc, sentiments_loc, scheme='resnik'):
        """
        Loads clusters obtained by using the serelex word net tool (http://serelex.cental.be).
        :param clusters_loc: folder containing serelex cluster files (.scf)
        :param sentiments_loc: file containing the sentiments associated with a cluster name
        """
        super(SerelexCluster, self).__init__()
        if scheme not in ():
            logging.warn('Scheme %s is not part of the allowed measures!' % scheme)
        self.clusters = {}
        for name in listdir(clusters_loc):
            cluster_name = path.splitext(name)
            self.clusters[cluster_name] = []
            for line in open(path.join(clusters_loc, name)):
                self.clusters[cluster_name].append(line.split())
        self.sentiments = {}
        for line in open(sentiments_loc):
            self.sentiments[line.split('\t')[0]] = line.strip().split('\t')[1]
        for name in set(self.clusters.keys()) - set(self.sentiments.keys()):
            logging.warn('No sentiment weight for cluster %s found!' % name)

    def get_score(self, tweet):
        return 0 or self or tweet

    def apply_filter(self, tweet, label):
        if not self.ranges:
            raise RuntimeError("Tried to run filter without specified ranges!")
        score = self.get_score(tweet)
        return self.ranges[str(label)][0] < score < self.ranges[str(label)][1]

    def apply_weighting(self, tweet, distances):
        if self.weight < 0 or self.pos_idx < 0 or self.neg_idx < 0:
            raise RuntimeError("Tried to run weighting without specified data!")
        score = self.get_score(tweet)
        if score > 0:
            distances[self.pos_idx] += self.weight * score
        elif score < 0:
            distances[self.neg_idx] -= self.weight * score


def analyse_mutator(mutator, latex=True):
    """
    Runs a raw weighting scheme for the specified mutator and prints
    interesting data
    :param mutator: SentMutator instance which is about to be analysed
    :param latex: if true, the output is printed latex pgf plot conform
    """
    mutator.add_weight(1, (1, 2))
    frq_pos = {}
    frq_neg = {}
    corpus_length = 0
    for line in open(root+'/Corpora/batches/tokenized.tsv', 'r', 'utf-8'):
        corpus_length += 1
        distances = mutator.apply_weighting(line.split('\t')[0], [0, 0])
        if distances[0] not in frq_pos:
            frq_pos[distances[0]] = 1
        else:
            frq_pos[distances[0]] += 1

        if distances[1] not in frq_neg:
            frq_neg[distances[1]] = 1
        else:
            frq_neg[distances[1]] += 1
    frq_pos.pop(0)
    frq_neg.pop(0)

    logging.info('\npositive distribution:')
    pos_dist = bucket_dist(frq_pos, 25)
    if latex:
        count = 0
        tmp = '{'
        for value in sorted(pos_dist.keys()):
            count += 1
            if count % 5 == 0:
                tmp += '\n'
            tmp += '(%.4f,%i) ' % (value, pos_dist[value])
        logging.info(tmp[:-1]+'}')
    else:
        for value in sorted(pos_dist.keys()):
            logging.info('%s: %s' % (value, pos_dist[value]))

    logging.info('\nnegative distribution:')
    neg_dist = bucket_dist(frq_neg, 25)
    if latex:
        count = 0
        tmp = '{'
        for value in sorted(neg_dist.keys()):
            count += 1
            if count % 5 == 0:
                tmp += '\n'
            tmp += '(%.4f,%i) ' % (value, neg_dist[value])
        logging.info(tmp[:-1]+'}')
    else:
        for value in sorted(neg_dist.keys()):
            logging.info('%s: %s' % (value, neg_dist[value]))

    logging.info('pos min key: %f, pos max key: %f' % (min(frq_pos.keys()), max(frq_pos.keys())))
    logging.info('pos min val: %i, pos max val: %i' % (min(frq_pos.values()), max(frq_pos.values())))
    pos_concat = [e for bucket in [([key]*val) for key, val in frq_pos.items()] for e in bucket]
    logging.info('pos median: %f' % (pos_concat[len(pos_concat)/2]))
    logging.info('pos avg: %f\n' % (sum(pos_concat)/len(pos_concat)))

    logging.info('neg min key: %f, neg max key: %f' % (min(frq_neg.keys()), max(frq_neg.keys())))
    logging.info('neg min val: %i, neg max val: %i' % (min(frq_neg.values()), max(frq_neg.values())))
    neg_concat = [e for bucket in [([key]*val) for key, val in frq_neg.items()] for e in bucket]
    logging.info('neg median: %f' % (neg_concat[len(neg_concat)/2]))
    logging.info('neg avg: %f\n' % (sum(neg_concat)/len(neg_concat)))

    logging.info('total average sentiment weight in corpus per tweet: %f'
                 % ((sum(pos_concat) + sum(neg_concat)) / corpus_length))


if __name__ == '__main__':
    clf = AutoCluster('/Users/Ceca/Arne/Data/logs/cluster.txt', '/Users/Ceca/Arne/Data/cluster_annotation/auto.txt')
