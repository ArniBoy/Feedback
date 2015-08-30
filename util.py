#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Arne'

import logging
import os
import re
from codecs import open
from subprocess import Popen, PIPE
from random import sample

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV

from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import Pipeline, FeatureUnion

from preprocessing import get_tweet, parse
from feature_extractors import AllCapsFeatures, HashtagFeatures, \
    PunctuationFeatures, ElongatedFeatures, EmoticonFeatures, SENTFeatures, \
    NRCFeatures, ItemSelector, DataSeparator, PosFeatures, LengthFeatures

lmt = WordNetLemmatizer()
root = '/home/arne/MasterArbeit/'
html_dict = {u'&quot;': u'"', u'&amp;': u'&', u'&lt;': u'<', u'&gt;': u'>',
             u'&OElig;': u'Œ', u'&oelig;': u'œ', u'&Scaron;': u'Š',
             u'&scaron;': u'š', u'&Yuml;': u'Ÿ', u'&circ;': u'ˆ',
             u'&tilde;': u'˜', u'&ndash;': u'–', u'&mdash;': u'—',
             u'&lsquo;': u'‘', u'&rsquo;': u'’', u'&sbquo;': u'‚',
             u'&ldquo;': u'“', u'&rdquo;': u'”', u'&bdquo;': u'„',
             u'&dagger;': u'†', u'&Dagger;': u'‡', u'&permil;': u'‰',
             u'&lsaquo;': u'‹', u'&rsaquo;': u'›', u'&euro;': u'€}'}


def init_logging():
    """
    Call at the beginning of each source file to ensure identical output options
    """
    setting = '[%(levelname)s %(filename)s:%(lineno)s\t- %(funcName)5s()]\t%(message)s'
    logging.basicConfig(format=setting, level=logging.INFO)


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


def get_raw_semeval_data(text, meta):
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
        sent, tweet, pos = get_tweet(u'\t'.join((sent, tweet, pos)))
        tweets[u'\t'.join([tweet, pos])] = sent
    return tweets.keys(), tweets.values()


def f1(y_true, y_pred, label):
    """
    Computes the isolated f1 and accuracy score for the specified label
    :param y_true: True labels
    :param y_pred: Predicted labels obtained from a model
    :param label: Label in question. One of preprocessing.POS/NEG/NEU/OBJ/OON
    :return: Harmonic mean between precision and recall, aka f1 score
    """
    tp = 0.0  # true positives
    fp = 0.0  # false positives
    fn = 0.0  # false negatives
    tn = 0.0  # true negatives
    for labels in zip(y_true, y_pred):
        if labels[1] == label:
            if labels[0] == label:
                tp += 1
            else:
                fp += 1
        else:
            if labels[0] == label:
                fn += 1
            else:
                tn += 1

    if tp == 0 or fp == 0 or fn == 0:
        micro_f1 = 0
    else:
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        micro_f1 = 2 * ((precision*recall) / (precision+recall))
    micro_accuracy = (tp + tn) / (tp + tn + fp + fn)
    return micro_f1, micro_accuracy


def precision(y_true, y_pred, label):
    tp = 0.0  # true positives
    fp = 0.0  # false positives
    for labels in zip(y_true, y_pred):
        if labels[1] == label:
            if labels[0] == label:
                tp += 1
            else:
                fp += 1

    if tp == 0 and fp == 0:
        return 0
    else:
        return tp / (tp+fp)


def get_final_semeval_data(classes, train_loc, dev_loc, test_loc):
    """
    Loads the final version of the semeval data that was merged by the following criteria:
    train: train13, dev13, train14, train15
    dev: test13, dev14, dev15
    test: test14, test15
    It should be noted that some redundancy exists between each year's set
    :param classes: bitflags following the definitions in preprocessing
    :param train_loc: location of training data
    :param dev_loc: location of development data
    :param test_loc: location of test data
    :return: tuple of lists, following the (X, y) scheme
    """
    train_labels, train_tweets, train_pos = parse(
        train_loc, classes
    )
    dev_labels, dev_tweets, dev_pos = parse(
        dev_loc, classes
    )
    test_labels, test_tweets, test_pos = parse(
        test_loc, classes
    )
    train = [e[0]+'\t'+e[1] for e in zip(train_tweets, train_pos)], train_labels
    dev = [e[0]+'\t'+e[1] for e in zip(dev_tweets, dev_pos)], dev_labels
    test = [e[0]+'\t'+e[1] for e in zip(test_tweets, test_pos)], test_labels
    return train, dev, test


def get_feature_union():
    """
    Builds the feature union used in all classifiers in this project
    :return: feature union of all implemented features
    """
    return FeatureUnion(
        transformer_list=[
            ('length', Pipeline([
                ('selector', ItemSelector(key='tweets')),
                ('counter', LengthFeatures()),
            ])),
            ('word_ngram', Pipeline([
                ('selector', ItemSelector(key='tweets')),
                ('tfidf', TfidfVectorizer()),
            ])),
            ('char_ngram', Pipeline([
                ('selector', ItemSelector(key='tweets')),
                ('tfidf', TfidfVectorizer()),
            ])),
            ('all_caps_count', Pipeline([
                ('selector', ItemSelector(key='tweets')),
                ('counter', AllCapsFeatures()),
            ])),
            ('hashtag_count', Pipeline([
                ('selector', ItemSelector(key='tweets')),
                ('counter', HashtagFeatures()),
            ])),
            ('punctuation_count', Pipeline([
                ('selector', ItemSelector(key='tweets')),
                ('counter', PunctuationFeatures()),
            ])),
            ('elongated_count', Pipeline([
                ('selector', ItemSelector(key='tweets')),
                ('counter', ElongatedFeatures()),
            ])),
            ('emoticon_count', Pipeline([
                ('selector', ItemSelector(key='tweets')),
                ('counter', EmoticonFeatures()),
            ])),
            ('pos', Pipeline([
                ('selector', ItemSelector(key='pos')),
                ('counter', PosFeatures()),
            ])),
            ('sent', Pipeline([
                ('selector', ItemSelector(key='tweets')),
                ('lex', SENTFeatures(root+'Data/Sentiment140-Lexicon-v0.1')),
            ])),
            ('nrc', Pipeline([
                ('selector', ItemSelector(key='tweets')),
                ('lex', NRCFeatures(root+'Data/NRC-Hashtag-Sentiment-Lexicon-v0.1')),
            ]))
        ],
        transformer_weights={
            'word_ngram': 1.0,
            'char_ngram': 1.0,
            'all_caps_count': 0.2,
            'hashtag_count': 0.2,
            'punctuation_count': 0.2,
            'elongated_count': 0.2,
            'emoticon_count': 0.2,
            'pos': 0.2,
            'sent': 0.05,
            'nrc': 0.05,
            'length': 0.2
        }
    )


def svm_pipeline(threads=-1):
    """
    Builds an untrained support vector machine using all implemented features.
    This is the main classifier, which is why the parameters are estimated by
    a -- now constant -- cross validated grid search.
    """
    pipeline = Pipeline([
        ('separate', DataSeparator()),
        ('features', get_feature_union()),
        ('svm', LinearSVC())
    ])
    parameters = {
        'features__word_ngram__tfidf__analyzer': ['word'],
        'features__word_ngram__tfidf__max_df': [0.75],
        'features__word_ngram__tfidf__ngram_range': [(1, 5)],
        'features__char_ngram__tfidf__analyzer': ['char'],
        'features__char_ngram__tfidf__max_df': [0.75],
        'features__char_ngram__tfidf__ngram_range': [(1, 5)],
        'svm__C': [0.5]
    }
    return GridSearchCV(pipeline, parameters, n_jobs=threads, verbose=1)


def k_means_pipeline(k):
    """
    Builds an untrained k-means model using all implemented features
    :param k: amount of clusters
    """
    return Pipeline([
        ('separate', DataSeparator()),
        ('features', get_feature_union()),
        ('k_means', MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1, init_size=1000, batch_size=1000))
    ])


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


def get_corpus(num_samples):
    """
    Load a subset of the sent 140 corpus. This function draws a random portion,
    as the original data seems to have a bias on position
    :param num_samples: a random sample of this size will be extracted. A negative value implies that
    everything get offered instad of just a section
    :return: The subset, as list of <tweet> tab <pos-tags> entries
    """
    all_data = [tweet.strip() for tweet in open(root+'Data/Corpora/batches/tokenized.tsv', encoding='utf-8')]
    if num_samples < 0:
        all_data = sample(all_data, len(all_data))
    else:
        all_data = sample(all_data, num_samples)
    return [u'\t'.join(get_tweet(tweet)[1:]) for tweet in all_data]


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


def analyse_auto_cluster(location):
    """
    Makes a frequency distribution of cluster labels in order to rebuild it at a later point
    :param location: Location of the cluster file .txt
    """
    clusters = {}
    for line in open(location):
        line = int(line.strip())
        if line in clusters:
            clusters[line] += 1
        else:
            clusters[line] = 1

    for key in clusters.keys():
        logging.info('%i: %i' % (key, clusters[key]))


def levenshtein(s1, s2):
    """
    Computes the levenshtein distance between two strings  s1 and s2. Implementation based on
    https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    :param s1: first string, the longer one. Will be adjusted if necessary.
    :param s2: second string, the shorter one. Will be adjusted if necessary.
    :return: modified levenshtein distance, where 'one' implies identical strings, and lower values less overlap, with
    'zero' meaning complete difference.
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    s1 = s1.lower()
    s2 = s2.lower()
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return 1 - (previous_row[-1] / float(len(s1)))


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


def wn_pos(universal_tag):
    """
    Parse the par of speech tags generated by the ARK tool to wordnet tags
    :param universal_tag: universal tag
    :return: wordnet tag, or empty if not interesting
    """
    if universal_tag.startswith('A'):
        return ADJ
    elif universal_tag.startswith('V'):
        return VERB
    elif universal_tag in ('N', 'S', 'Z', 'L', 'M'):
        return NOUN
    elif universal_tag.startswith('R'):
        return ADV
    else:
        return ''


def get_wordnet_wordlist(source, sink):
    """
    Extracts all unique words from a wordnet distribution, located as 'data.part-of-speech' files in a 'dict' folder
    :param source: target location
    :param sink: write location
    """
    words = set()
    for wordlist in [name for name in os.listdir(source) if name.startswith('data.')]:
        for line in open(os.path.join(source, wordlist)):
            if not line.startswith(' '):
                word = line.split(' ')[4].lower()
                for pos in (NOUN, VERB, ADJ, ADV):
                    words.add(lmt.lemmatize(word, pos))
    with open(sink, 'w') as write_loc:
        for word in words:
            write_loc.write('%s\n' % word)


def get_wordnet_pairs(wordlist, text, tags, cluster):
    """
    Generates all possible pairs between a tweet and word-clusters
    :param wordlist: set of all terms in wordnet, simply speeds up the process by pre-filtering the words in the tweet
    :param text: sequence of words
    :param tags: sequence of part of speech tags, one per word in text
    :param cluster: word list associated with a sentient
    :return: possible pairs between the tweet, cluster, and wordnet. To be used with wordnet::similarity
    """
    ret = []
    split_text = text.strip().split(' ')
    split_tags = [wn_pos(tag) for tag in tags.strip().split(' ')]
    stemmed_tweet = {lmt.lemmatize(t, p): p for t, p in zip(split_text, split_tags) if p}
    logging.debug('Target tweet contains a total of %i unique words' % len(stemmed_tweet))
    for t_word, pos in stemmed_tweet.items():
        if pos and t_word in wordlist:
            for c_word in cluster:
                ret.append('%s#%s %s#%s' % (t_word, pos, c_word, pos))
    logging.debug('Total overlap with wordnet: %i' % (len(ret)/4/len(cluster)))
    return ret


def get_serelex_cluster(keyword, target, top_n=100):
    """
    Collects a word cluster from the serelex cluster implementation. The result is transformed into a word list at the
    target location, lemmatized and sorted for impact.
    :param keyword: cluster seed
    :param target: target directory for cluster dumping
    :param top_n: maximum amount of words saved to target
    """
    pattern = re.compile(r'"word": "(.*?)",')
    call = Popen(['curl', 'http://serelex.cental.be/find/norm60-corpus-all/%s' % keyword], stdout=PIPE)
    words = {}
    counter = 0
    for match in pattern.findall(''.join(call.stdout.read())):
        if match.count(' '):
            continue
        counter += 1
        for pos in (NOUN, VERB, ADJ, ADV):
            if len(words) >= top_n:
                break
            word = lmt.lemmatize(match, pos).lower()
            if word not in words:
                words[word] = counter
    words = [word for word, count in sorted(words.items(), key=lambda x: x[1])]
    logging.info('A total of %i words were collected for the "%s" cluster' % (len(words), keyword))
    with open(os.path.join(target, keyword), 'w') as sink:
        for word in words:
            sink.write('%s\n' % word)


if __name__ == '__main__':
    init_logging()
