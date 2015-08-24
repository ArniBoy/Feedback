#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Arne Recknagel'

import logging
from codecs import open
import re

# possible sentiment labels used as flags
POS = 1
NEG = 2
NEU = 4
OBJ = 8
OON = 16

# 'label transform dictionary'
ltd = {u'positive': POS,
       u'negative': NEG,
       u'neutral': NEU,
       u'objective': OBJ,
       u'objective-OR-neutral': OON}

# regular expressions
neg_re = re.compile(r"(?:"
                    r"^(?:never|no|nothing|nowhere|noone|none|not|"
                    r"havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|"
                    r"wouldnt|dont|doesnt|didnt|isnt|arent|aint)$"
                    r")|n't")
punct_re = re.compile(r"[\.,:;!\?]")
elong_re = re.compile(r"(.)\1{2,}")


def pp_negation(tweet):
    """
    Negation as defined in http://sentiment.christopherpotts.net/lingstruc.html,
    but also includes ',' as context finisher.
    :param tweet:
    :return: Tweet with altered negated context.
    """
    neg = False
    tokens = tweet.split(u' ')
    for idx in range(len(tokens)):
        if neg_re.search(tokens[idx]):
            neg = True
            continue
        if punct_re.search(tokens[idx]):
            neg = False
            continue
        if neg:
            tokens[idx] += u'_NEG'
    return u' '.join(tokens)


def pp_elongation(tweet):
    """
    Constrains the number of elongated characters to three.
    :param tweet:
    :return: Normalized tweet.
    """
    tokens = tweet.split(u' ')
    for idx in range(len(tokens)):
        tokens[idx] = elong_re.sub(r"\1\1\1", tokens[idx])
    return u' '.join(tokens)


def pp_urls(tweet):
    """
    Normalize urls in a tweet.
    :param tweet: Input tweet
    :return: Normalized tweet.
    """
    tokens = tweet.split(u' ')
    for idx in range(len(tokens)):
        if tokens[idx].startswith(u'http'):
            tokens[idx] = u'http://url'
    return u' '.join(tokens)


def pp_users(tweet):
    """
    Normalize user mentions in a tweet.
    :param tweet: Input tweet
    :return: Normalized tweet.
    """
    tokens = tweet.split(u' ')
    for idx in range(len(tokens)):
        if tokens[idx].startswith(u'@'):
            tokens[idx] = u'@ref'
    return u' '.join(tokens)


def get_tweet(tweet):
    """
    Chomps a tweet with POS tags to relevant data and starts the preprocessing
    chain. If a label is given, it gets passed as well.
    :param tweet: full line, including tweet and user id, as well as tweet
    sentiment and raw tweet content
    :return: label, tweet, pos-tags
             -as-
             unicode, unicode, unicode
    """
    data = tweet.strip().split(u'\t')
    if len(data) == 2:
        t, p = data
        l = None
    else:
        l, t, p = data
    if t != u'Not Available':
        t = pp_urls(t)        # normalizes url mentions
        t = pp_users(t)       # normalizes user mentions
        t = pp_elongation(t)  # normalizes character elongations
        # t = pp_negation(t)    # adds negated context (atm bad for performance)
    return l, t, p


def read_tweet_file(f_loc, flags):
    """
    Read a file of tab-separated tweets, at this stage with labeled data.
    :param f_loc: twitterData/twitter-train-full-B.tsv
                  -or-
                  twitterData/tweeter-dev-full-B.tsv
    :param flags: Bitwise flags
    :return: labels, content, pos-tags
             -as generator of-
             list(unicode), list(unicode), list(unicode)
    """
    count = 0
    for entry in open(f_loc, 'r', 'utf-8'):
        label, tweet, pos = get_tweet(entry)
        if tweet != u'Not Available' and (ltd[label] & flags):
            count += 1
            yield ltd[label], tweet, pos
    else:
        logging.debug('Parsed file: %s with a total of %i entries for flags %s'
                      % (f_loc, count, bin(flags)))


def parse(f_loc, flags=0):
    """
    Wrapper for a flattened output of read_tweet_file, also defaults to shuffled
    content for better testing.
    :param f_loc: twitterData/twitter-train-full-B.tsv
                  -or-
                  twitterData/tweeter-dev-full-B.tsv
    :param flags: Bitwise flags
    :return: labels, content, pos-tags
             -as-
             list(unicode), list(unicode), list(unicode)
    """
    labels, tweets, pos = map(list, zip(*list(read_tweet_file(f_loc, flags))))
    return labels, tweets, pos
