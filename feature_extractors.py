#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Arne Recknagel'

import logging
import os
import re
import gzip
import numpy
import threading
from os.path import join

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class DataSeparator(BaseEstimator, TransformerMixin):
    """Separates each line of data by '\t' and sorts them according to their
     type. Currently, that is {tweets, pos}. Each slice of data can be accessed
     via its keyword in all following Transformers in the pipeline.
    """
    def fit(self, x, y=None):
        return self

    def transform(self, data):
        features = {'tweets': [], 'pos': []}
        for entry in data:
            tweet, pos = entry.strip().split('\t')
            features['tweets'].append(tweet)
            features['pos'].append(pos)
        return features


class ItemSelector(BaseEstimator, TransformerMixin):
    """ Author: Matt Terry <matt.terry@gmail.com>, see
    http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
    For data grouped by feature, select subset of data at a provided key.
    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.
    >> len(data[key]) == n_samples

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class TfidfVectorizerWrapper(TfidfVectorizer):
    """
    Wrapper for the sklearn class. Obsolete now, since I use a a data separator
    and item selector to chose between input data sections. But this is how to
    overload sklearn transformers, so it stays.
    """
    def transform(self, tweets, copy=True):
        #  tweets = [tweet.lower() for tweet in tweets]
        return TfidfVectorizer.transform(self, tweets, copy)


class LexFeatures(BaseEstimator, TransformerMixin):
    """
    This base-class is used as dictionary loader, implementing sub-classes are
    supposed to implement the 'transform' function and use at least one of them.
    sklearn uses multiprocessing to speed up some processes (indicated by the
    n_jobs keyword being greater than 1), so each process might load an own copy
    of the lexicons. Sharing data across processes is hard, so unless the
    lexicons get huge I don't feel like the current format needs to change.
    Sentiment lexicon format:
        term<tab>sentimentScore<tab>numPositive<tab>numNegative
    Internal representation:
        dict(term -> (score, num_pos, num_neg))
    """
    nrc_lock = threading.Lock()
    snt_lock = threading.Lock()
    nrc_is_init = False
    snt_is_init = False
    nrc_hashtag_uni = {}
    sent_140_uni = {}
    nrc_hashtag_bi = {}
    sent_140_bi = {}
    nrc_hashtag_pair = {}
    sent_140_pair = {}

    @staticmethod
    def load_nrc(nrc_dir):
        LexFeatures.nrc_lock.acquire()
        if not LexFeatures.nrc_is_init:
            logging.debug('Process %i: Loading NRC lexicon..' % os.getpid())
            with gzip.open(join(nrc_dir, 'unigrams-pmilexicon.txt.gz')) \
                    as nrc_uni_lex:
                for line in nrc_uni_lex:
                    tmp = line.strip().split('\t')
                    LexFeatures.nrc_hashtag_uni[tmp[0]] = \
                        (float(tmp[1]), int(tmp[2]), int(tmp[3]))
            with gzip.open(join(nrc_dir, 'bigrams-pmilexicon.txt.gz')) \
                    as nrc_bi_lex:
                for line in nrc_bi_lex:
                    tmp = line.strip().split('\t')
                    LexFeatures.nrc_hashtag_bi[tmp[0]] = \
                        (float(tmp[1]), int(tmp[2]), int(tmp[3]))
            with gzip.open(join(nrc_dir, 'pairs-pmilexicon.txt.gz')) \
                    as nrc_skip_lex:
                for line in nrc_skip_lex:
                    tmp = line.strip().split('\t')
                    LexFeatures.nrc_hashtag_pair[tmp[0]] = \
                        (float(tmp[1]), int(tmp[2]), int(tmp[3]))
            LexFeatures.nrc_is_init = True
            logging.debug('done.')
        LexFeatures.nrc_lock.release()

    @staticmethod
    def load_snt(snt_dir):
        LexFeatures.snt_lock.acquire()
        if not LexFeatures.snt_is_init:
            logging.debug('Process %i: Loading SENT lexicon..' % os.getpid())
            with gzip.open(join(snt_dir, 'unigrams-pmilexicon.txt.gz')) \
                    as sent_uni_lex:
                for line in sent_uni_lex:
                    tmp = line.strip().split('\t')
                    LexFeatures.sent_140_uni[tmp[0]] = \
                        (float(tmp[1]), int(tmp[2]), int(tmp[3]))
            with gzip.open(join(snt_dir, 'bigrams-pmilexicon.txt.gz')) \
                    as sent_bi_lex:
                for line in sent_bi_lex:
                    tmp = line.strip().split('\t')
                    LexFeatures.sent_140_bi[tmp[0]] = \
                        (float(tmp[1]), int(tmp[2]), int(tmp[3]))
            with gzip.open(join(snt_dir, 'pairs-pmilexicon.txt.gz')) \
                    as sent_skip_lex:
                for line in sent_skip_lex:
                    tmp = line.strip().split('\t')
                    LexFeatures.sent_140_pair[tmp[0]] = \
                        (float(tmp[1]), int(tmp[2]), int(tmp[3]))
            LexFeatures.snt_is_init = True
            logging.debug('done.')
        LexFeatures.snt_lock.release()

    def fit(self, x, y=None):
        return self

    def scores(self, sent_dict, tokens):
        """
        Lexical feature scoring function, as defined in Mohammad et. al. (2014)
        :param sent_dict: Sentiment dictionary
        :param tokens: Token list, containing alle uni-, bi-, and skip-gram found
        in the target tweet
        :return: tuple of 4 scores
        """
        pos_count = 0
        sum_score = 0
        max_score = 0
        last_pos = 0
        for token in tokens:
            sent = sent_dict.get(token, None)
            if sent is not None:
                pos_count += 1 if sent[0] > 0 else 0
                sum_score += sent[0]
                max_score = sent[0] if sent[0] > max_score else max_score
                last_pos = sent[0] if sent[0] > 0 else last_pos
        return pos_count, sum_score, max_score, last_pos

    def transform(self, tweets):
        NotImplementedError("Class %s doesn't implement " %
                            (self.__class__.__name__,))


class NRCFeatures(LexFeatures):
    """
    Uses the nrc lexicon to build lexical features for a range of tweets.
    """
    def __init__(self, nrc_dir):
        self.nrc_dir = nrc_dir

    def transform(self, tweets):
        LexFeatures.load_nrc(self.nrc_dir)
        features = numpy.zeros((len(tweets), 12))
        for count, tweet in enumerate(tweets):
            # make tweet n-gram and pair lists
            uni_grams = tweet.split(u' ')
            bi_grams = [tweet[i]+u' '+tweet[i+1] for i in range(len(tweet)-1)]
            u_idx = range(len(uni_grams))
            b_idx = range(len(bi_grams))
            pairs = [uni_grams[i]+u'---'+uni_grams[j]
                     for i in u_idx for j in u_idx if i != j]
            pairs.extend([bi_grams[i]+u'---'+bi_grams[j]
                         for i in b_idx for j in b_idx if i != j])
            pairs.extend([uni_grams[i]+u'---'+bi_grams[j]
                         for i in u_idx for j in b_idx])
            pairs.extend([bi_grams[i]+u'---'+uni_grams[j]
                         for i in b_idx for j in u_idx])

            # uni-gram sentiment features
            features[count][0], features[count][1], features[count][2], features[count][3] = \
                self.scores(LexFeatures.nrc_hashtag_uni, uni_grams)

            # bi-gram sentiment features
            features[count][4], features[count][5], features[count][6], features[count][7] = \
                self.scores(LexFeatures.nrc_hashtag_bi, bi_grams)

            # pair sentiment features
            features[count][8], features[count][9], features[count][10], features[count][11] = \
                self.scores(LexFeatures.nrc_hashtag_pair, pairs)
        return features


class SENTFeatures(LexFeatures):
    """
    Uses the sent lexicon to build lexical features for a range of tweets.
    """
    def __init__(self, snt_dir):
        self.snt_dir = snt_dir

    def transform(self, tweets):
        LexFeatures.load_snt(self.snt_dir)
        features = numpy.zeros((len(tweets), 12))
        for count, tweet in enumerate(tweets):
            # make tweet n-gram and pair lists
            uni_grams = tweet.split(u' ')
            bi_grams = [tweet[i]+u' '+tweet[i+1] for i in range(len(tweet)-1)]
            u_idx = range(len(uni_grams))
            b_idx = range(len(bi_grams))
            pairs = [uni_grams[i]+u'---'+uni_grams[j]
                     for i in u_idx for j in u_idx if i != j]
            pairs.extend([bi_grams[i]+u'---'+bi_grams[j]
                         for i in b_idx for j in b_idx if i != j])
            pairs.extend([uni_grams[i]+u'---'+bi_grams[j]
                         for i in u_idx for j in b_idx])
            pairs.extend([bi_grams[i]+u'---'+uni_grams[j]
                         for i in b_idx for j in u_idx])

            # uni-gram sentiment features
            features[count][0], features[count][1], features[count][2], features[count][3] = \
                self.scores(LexFeatures.sent_140_uni, uni_grams)

            # bi-gram sentiment features
            features[count][4], features[count][5], features[count][6], features[count][7] = \
                self.scores(LexFeatures.sent_140_bi, bi_grams)

            # pair sentiment features
            features[count][8], features[count][9], features[count][10], features[count][11] = \
                self.scores(LexFeatures.sent_140_pair, pairs)
        return features


class AllCapsFeatures(BaseEstimator, TransformerMixin):
    """
    Mini feature, gives the number of allcaps words in a tweet.
    """
    def fit(self, x, y=None):
        return self

    def transform(self, tweets):
        features = numpy.zeros((len(tweets), 1))
        for count, tweet in enumerate(tweets):
            for word in tweet.split(u' '):
                if word != u'I' and word.isupper():
                    features[count, 0] += 1
        return features


class LengthFeatures(BaseEstimator, TransformerMixin):
    """
    Mini feature, some length related information for the tweet.
    """
    def fit(self, x, y=None):
        return self

    def transform(self, tweets):
        features = numpy.zeros((len(tweets), 3))
        for count, tweet in enumerate(tweets):
            for word in tweet.split(u' '):
                features[count, 0] += 1
                word_length = len(word)
                features[count, 1] += word_length
                features[count, 2] = word_length if word_length > features[count, 2] else features[count, 2]
            features[count, 1] /= float(features[count, 0])
        return features


class HashtagFeatures(BaseEstimator, TransformerMixin):
    """
    Mini feature, gives number of hasthags in a tweet.
    """
    def fit(self, x, y=None):
        return self

    def transform(self, tweets):
        features = numpy.zeros((len(tweets), 1))
        for count, tweet in enumerate(tweets):
            for word in tweet.split(u' '):
                if word.startswith(u'#'):
                    features[count, 0] += 1
        return features


class ElongatedFeatures(BaseEstimator, TransformerMixin):
    """
    Mini feature, gives the number of elongated words in a tweet.
    """
    elong_re = re.compile(r"(.)\1{2,}")

    def fit(self, x, y=None):
        return self

    def transform(self, tweets):
        features = numpy.zeros((len(tweets), 1))
        for count, tweet in enumerate(tweets):
            for word in tweet.split(u' '):
                if ElongatedFeatures.elong_re.search(word):
                    features[count, 0] += 1
        return features


class EmoticonFeatures(BaseEstimator, TransformerMixin):
    """
    Mini feature, gives the number of smileys and if the last token in the tweet
    was a smiley.
    TODO: Add positive/negative list of smileys, return
    (pos_count, neg_count, 1 if last_pos==pos -1 if last_pos==neg else 0)
    instead of current output
    """
    emoticon_re = re.compile(r"(?:"
                             r"[<>]?"
                             r"[:;=8]"
                             r"[\-o\*\']?"
                             r"[\)\]\(\[dDpP/\\:\}\{@\|\\]"
                             r"|"
                             r"[\)\]\(\[dDpP/\\:\}\{@\|\\]"
                             r"[\-o\*\']?"
                             r"[:;=8]"
                             r"[<>]?"
                             r")")

    def fit(self, x, y=None):
        return self

    def transform(self, tweets):
        features = numpy.zeros((len(tweets), 2))
        for count, tweet in enumerate(tweets):
            words = tweet.split(u' ')
            for word in words:
                if EmoticonFeatures.emoticon_re.search(word):
                    features[count, 0] += 1
            if EmoticonFeatures.emoticon_re.search(words[-1]):
                features[count, 1] = 1
        return features


class PunctuationFeatures(BaseEstimator, TransformerMixin):
    """
    Mini feature, gives number of multiple question marks, number of multiple
    exclamation marks, number of multiple mixes marks and if the last sign in
    the last token is a mark.
    """
    que = re.compile(r'\?{2,}')
    exc = re.compile(r'!{2,}')
    quexc = re.compile(r'[!\?]{2,}')

    def fit(self, x, y=None):
        return self

    def transform(self, tweets):
        features = numpy.zeros((len(tweets), 4))
        for count, tweet in enumerate(tweets):
            words = tweet.split(u' ')
            for word in words:
                if PunctuationFeatures.que.search(word):
                    features[count, 0] += 1
                if PunctuationFeatures.exc.search(word):
                    features[count, 1] += 1
                if PunctuationFeatures.quexc.search(word):
                    features[count, 2] += 1
            if u'!' in words[-1] or u'?' in words[-1]:
                features[count, 3] = 1
        return features


class PosFeatures(BaseEstimator, TransformerMixin):
    """
    Gives a simple count vector for part of speech tags.
    """
    tags = {'N': 0, 'O': 1, '^': 2, 'S': 3, 'Z': 4, 'V': 5, 'A': 6, 'R': 7,
            '!': 8, 'D': 9, 'P': 10, '&': 11, 'T': 12, 'X': 13, '#': 14,
            '@': 15, '~': 16, 'U': 17, 'E': 18, '$': 19, ',': 20, 'G': 21,
            'L': 22, 'M': 23, 'Y': 24}

    def fit(self, x, y=None):
        return self

    def transform(self, pos):
        features = numpy.zeros((len(pos), len(PosFeatures.tags)))
        for count_x, line in enumerate(pos):
            for idx in range(0, len(line), 2):
                features[count_x, PosFeatures.tags[line[idx]]] += 1
        return features