#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Arne'

import logging
import os
from codecs import open
from subprocess import call

from external_sources import AFinnWordList

root = '/Users/Ceca/Arne/Data/'
html_dict = {u'&quot;': u'"', u'&amp;': u'&', u'&lt;': u'<', u'&gt;': u'>',
             u'&OElig;': u'Œ', u'&oelig;': u'œ', u'&Scaron;': u'Š',
             u'&scaron;': u'š', u'&Yuml;': u'Ÿ', u'&circ;': u'ˆ',
             u'&tilde;': u'˜', u'&ndash;': u'–', u'&mdash;': u'—',
             u'&lsquo;': u'‘', u'&rsquo;': u'’', u'&sbquo;': u'‚',
             u'&ldquo;': u'“', u'&rdquo;': u'”', u'&bdquo;': u'„',
             u'&dagger;': u'†', u'&Dagger;': u'‡', u'&permil;': u'‰',
             u'&lsaquo;': u'‹', u'&rsaquo;': u'›', u'&euro;': u'€}'}


def make_dir(d):
    """
    Wrapper to safely make target directory
    :param d: Directory to make
    """
    if not os.path.exists(d):
        os.makedirs(d)


def f_range(x, y, jump):
    """
    Emulates the native range function for floating point numbers
    :param x: start
    :param y: stop
    :param jump: step
    :return: generator of the sequence
    """
    while x <= y:
        yield x
        x += jump


def get_semeval_data(text, meta):
    """
    Extracting preprocessed semeval data, consisting of tweets and POS-tags as instances, and strings as labels
    :param text: location of tweet text
    :param meta: location of meta data
    :return: Training or test data in the classic (X, y) format
    """
    tweets = {}
    for line_1, line_2 in zip(open(text, 'r', 'utf-8'),
                              open(meta, 'r', 'utf-8')):
        tweet, pos, _, _ = line_1.strip().split('\t')
        _, _, sent = line_2.strip().split('\t')
        tweets[u'\t'.join([tweet, pos])] = sent
    return tweets.keys(), tweets.values()


def get_columns(doc, columns, delim, only_if=-1):
    """
    Extract columns from target document, write them to cleaned tsv file
    :param doc: Target document
    :param columns: Columns to be extracted
    :param delim: Column delimiter
    :param only_if: Ignore rows with different column count
    """
    with open(os.path.splitext(doc)[0]+'_cleaned.tsv', 'w') as sink:
        for line in open(doc):
            data = line.strip().split(delim)
            if 0 <= only_if == len(data) or only_if < 0:
                sub_row = ''
                for column in columns:
                    sub_row += '%s\t' % data[column]
                sink.write(sub_row[:-1] + '\n')


def batch_file(doc, batch_size=10000):
    """
    Split file into batches
    :param doc: Target document
    :param batch_size: Desired maximum batch size
    """
    d = os.path.join(os.path.dirname(doc), 'batches')
    make_dir(d)
    batch = []
    count = 0
    for line_a in open(doc):
        if len(batch) < batch_size:
            batch.append(line_a)
        else:
            with open(os.path.join(d, 'batch_%i.tsv' % count), 'w') as sink:
                for line_b in batch:
                    sink.write(line_b)
            batch = []
            count += 1
    else:
        with open(os.path.join(d, 'batch_%i.tsv' % count), 'w') as sink:
            for line_b in batch:
                sink.write(line_b)


def fix_html(doc):
    """
    Repairs html sub-symbols to original state -- needs to happen after column split, since " is part of the substituted
    symbols.
    :param doc: Original document, gets overwritten
    """
    fixed = []
    for line in open(doc, encoding='utf-8'):
        for key, val in html_dict.items():
            line = line.replace(key, val)
        fixed.append(line)
    with open(doc, 'w', encoding='utf-8') as sink:
        for line in fixed:
            sink.write(line)


def get_serelex_cluster(keyword):
    """
    Collects a word cluster from the serelex cluster implementation. No pre processing is done, so a raw json object as
    string is returned
    :param keyword: Central cluster word
    :return: Cluster, plus distance measures
    """
    return call(['curl', 'http://serelex.cental.be/find/norm60-corpus-all/%s' % keyword])


def analyse_auto_cluster(location):
    """
    Makes a frequency distribution of cluster labels in order to rebuild it at a later point
    :param location: Location of the cluster file .txt
    """
    clusters = {}
    for line in open(root+location):
        line = int(line.strip())
        if line in clusters:
            clusters[line] += 1
        else:
            clusters[line] = 1

    for key in clusters.keys():
        logging.info('%i: %i' % (key, clusters[key]))


def bucket_dist(raw_dict, num_buckets):
    """
    Builds a discrete image from a frequency distribution, with a total of 'num_buckets' linearly distributed buckets
    :param raw_dict: original distribution
    :param num_buckets: granularity
    :return: bucketed form
    """
    min_val = min(raw_dict.keys())
    max_val = max(raw_dict.keys())
    dictionary = dict.fromkeys(f_range(min_val, max_val*1.001, (max_val-min_val) / num_buckets), 0)
    for key in raw_dict.keys():
        for bucket in sorted(dictionary.keys()):
            if key <= bucket:
                dictionary[bucket] += raw_dict[key]
                break
    return dictionary


def analyse_mutator(corpus_loc, latex=True):
    """
    Runs a raw weighting scheme for the AFinn source and prints statistical data
    :param corpus_loc: location of the corpus that gets weighted
    :param latex: if true, the output is printed latex pgf plot conform
    """
    mutator = AFinnWordList(root+'/afinn/AFINN-111.txt')
    mutator.add_weight(1, (1, 2))
    frq_pos = {}
    frq_neg = {}
    corpus_length = 0
    for line in open(root+corpus_loc, 'r', 'utf-8'):
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
