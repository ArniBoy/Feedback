#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Arne Recknagel'

# basic library
import cPickle
import sys
import random
from itertools import permutations, combinations
from argparse import ArgumentParser
import logging
FORMAT = "[%(levelname)s %(filename)s:%(lineno)s\t- %(funcName)5s()]\t" \
         "%(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)

# sklearn modules
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from feature_extractors import ItemSelector, DataSeparator, PosFeatures,\
    AllCapsFeatures, HashtagFeatures, PunctuationFeatures, ElongatedFeatures,\
    EmoticonFeatures, SENTFeatures, NRCFeatures

# custom modules
import preprocessing
from backfeed import Feeder

parser = ArgumentParser(description="""
  This script can either be used to test the current classifier with its
  settings, or to start the backfeed routine.""")
parser.add_argument('model', help="""
  Path to either a trained model or the location where the newly trained model
  will be saved. Which options is chosen depends on whether '--load'/'-l' is
  set.""")
parser.add_argument('train', help="""Path to the training data.""")
parser.add_argument('dev', help="""Path to the development data.""")
parser.add_argument('nrc_dir', help="""
  The location of the NRC sentiment lexicon which is needed for the
  LexFeatures.""")
parser.add_argument('snt_dir', help="""
  The location of the Sentiment140 lexicon which is needed for the
  LexFeatures.""")
parser.add_argument('labels', help="""
  Defines the labels which will be gathered from the train/dev data, and by
  extension the classes which get classified in the backfeed loop. Labels can be
  selected flexibly, any combinations of {p for positive, n for negative, e for
  neutral, o for objective, r for objective or neutral} works. For example 'nep'
  for positive, negative, and neutral labels.""")
parser.add_argument('-l', '--load', action='store_true', help="""
  If set, the 'model' argument is used as source for the initially trained
  model. If not, the initial model will be saved there instead.""")
parser.add_argument('-c', '--corpus', type=str, help="""
  Path to the corpus used in the backfeed loop. If this field is set, the
  program assumes that a backfeed should be run. Omit this field for a single
  test.""")
parser.add_argument('-a', '--a_size', type=int, default=300, help="""
  Backfeed-only, ignored else. In each call of the backfeed, a number of corpus
  entries get added to the training data. This parameter controls the size of
  each addition.""")
parser.add_argument('-i', '--i_size', type=int, default=5, help="""
  Backfeed-only, ignored else. Controls the number of backfeed iterations""")


def init_model(nrc_dir, snt_dir):
    """
    Wrap feature extraction and the classificator into a pipeline, and wrap the
    pipeline with all desired parameter ranges into a grid search.
    :return: untrained parametrized model
    """
    features = FeatureUnion(
        transformer_list=[
            ('all_caps', Pipeline([
                ('selector', ItemSelector(key='tweets')),
                ('counter', AllCapsFeatures()),
            ])),
            ('lexical_sentiments_nrc', Pipeline([
                ('selector', ItemSelector(key='tweets')),
                ('counter', NRCFeatures(nrc_dir=nrc_dir)),
            ])),
            ('lexical_sentiments_sent', Pipeline([
                ('selector', ItemSelector(key='tweets')),
                ('counter', SENTFeatures(snt_dir=snt_dir)),
            ])),
            ('word_ngram', Pipeline([
                ('selector', ItemSelector(key='tweets')),
                ('tfidf', TfidfVectorizer()),
            ])),
            ('char_ngram', Pipeline([
                ('selector', ItemSelector(key='tweets')),
                ('tfidf', TfidfVectorizer()),
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
            ('pos_count', Pipeline([
                ('selector', ItemSelector(key='pos')),
                ('counter', PosFeatures()),
            ]))
        ],
        transformer_weights={
            'all_caps': 1.0,
            'lexical_sentiments_nrc': 1.0,
            'lexical_sentiments_sent': 1.0,
            'word_ngram': 1.0,
            'char_ngram': 1.0,
            'hashtag_count': 1.0,
            'punctuation_count': 1.0,
            'elongated_count': 1.0,
            'emoticon_count': 1.0,
            'pos_count': 1.0,
        }
    )
    pipeline = Pipeline([
        ('separate', DataSeparator()),
        ('features', features),
#        ('normalizer', StandardScaler()),
        ('svm', LinearSVC())
    ])
    parameters = {
        'features__word_ngram__tfidf__analyzer': ('word',),
        'features__word_ngram__tfidf__max_df': (0.75,),
        'features__word_ngram__tfidf__ngram_range': ((1, 2),),
        'features__char_ngram__tfidf__analyzer': ('char',),
        'features__char_ngram__tfidf__max_df': (0.75,),
        'features__char_ngram__tfidf__ngram_range': ((1, 4),),
#        'normalizer__with_mean': (False,),
        'svm__C': (range(1, 10))
    }
    return GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)


def f1(y_true, y_pred, label):
    """
    Computes the isolated f1 score for the specified label
    :param y_true: True labels
    :param y_pred: Predicted labels obtained from a model
    :param label: Label in question. One of preprocessing.POS/NEG/NEU/OBJ/OON
    :return: Harmonic mean between precision and recall, aka f1 score
    """
    tp = 0.0  # true positives
    fp = 0.0  # false positives
    fn = 0.0  # false negatives
    for labels in zip(y_true, y_pred):
        if labels[1] == label:
            if labels[0] == labels[1]:
                tp += 1
            else:
                fp += 1
            continue
        if labels[0] == label:
            fn += 1
    if tp == 0 and (fp == 0 or fn == 0):
        logging.warning('Calculation impossible with tp=%i, fp=%i, fn=%i for '
                        'label: %s' % (tp, fp, fn, label))
        label_data = ''
        for entry in zip(y_true, y_pred):
            label_data += '\n\tTrue: %s, Pred: %s' % entry
        logging.warning(label_data)
        exit(1)
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    micro_f1 = 2 * ((precision*recall) / (precision+recall))
    logging.debug('Micro f1 for label %s: %f' % (label, micro_f1))
    return micro_f1


def macro_avg_f1(y_true, y_pred, classes):
    """
    Macro averaged f1 score of all specified labels
    :param y_true: True labels
    :param y_pred: Predicted labels obtained from a model
    :param classes: Union of labels/classes in question
    :return: Simple average between all f1 scores of all give labels
    """
    scores = 0.0
    count = 0
    class_union = [preprocessing.POS, preprocessing.NEG, preprocessing.NEU,
                   preprocessing.OBJ, preprocessing.OON]
    for label in class_union:
        if label & classes:
            count += 1
            scores += f1(y_true, y_pred, label)
    return scores / count


"""def uniformify(x, y, rand=True):

    Given a list of data points and a list of corresponding labels, this
    function returns a version with equal amounts of labels among all data
    points. This can improve performance for svm training -- at the cost of
    tossing data away.
    :param x: The data point list
    :param y: The label list
    :param rand: If true, shuffle the resulting lists
    :return: x and y, uniformified

    new = []
    shortest = len(x)
    labels = set(y)
    for label in labels:
        tmp = [x[i] for i, elem in enumerate(y) if elem == label]
        new.append([tmp, [label]*len(tmp)])
        if len(tmp) < shortest:
            shortest = len(tmp)
    for count, entry in enumerate(new):
        entry[0] = entry[0][:shortest]
        entry[1] = entry[1][:shortest]
        new[count] = zip(entry[0], entry[1])
    new = [elem for sub_list in new for elem in sub_list]
    if rand:
        random.shuffle(new)
    logging.info('Removed a total of %i entries, %i left' %
                 (len(x)-len(new), len(new)))
    return map(list, zip(*new))"""


def print_scores(model, x_dev, y_dev, classes):
    """
    Print evaluation data from the currently trained model
    :param model: classifier, trained
    :param x_dev: dev data
    :param y_dev: dev labels
    :param classes: flag variable, see re_train description
    """
    predicted = model.predict(x_dev)
    s_f1 = f1_score(y_true=y_dev,
                    y_pred=predicted,
                    average='macro')
    s_acc = accuracy_score(y_true=y_dev,
                           y_pred=predicted)
    s_m = macro_avg_f1(y_dev, predicted, classes)
    logging.info('Current scores: '
                 '\n\tf1:       %f'
                 '\n\taccuracy: %f'
                 '\n\tmacro f1: %f' % (s_f1, s_acc, s_m))


def re_train(model, model_loc, train, dev, corpus_loc, num, num_runs, classes):
    """
    runs the back feed mechanism.
    :param model: classificator
    :param model_loc: If the model was not loaded, a raw one is trained and
    saved to this location
    :param train: File with annotated training tweets
    :param dev: File with annotated development tweets
    :param corpus_loc: Directory of the Sentiment140 corpus
    :param num: Amount of added corpus data per iteration
    :param num_runs: Number of iterations
    :param classes: Variable holding bit flags corresponding to class labels --
    definition is in preprocessing variable 'ltd'
    """
    # load training data, is needed either way
    train_labels, train_tweets, train_pos = preprocessing.parse(
        train, classes
    )
    x_train = [e[0]+'\t'+e[1] for e in zip(train_tweets, train_pos)]
    y_train = train_labels
    # x_train, y_train = uniformify(x_train, y_train)

    # if model is raw, train it
    if not hasattr(model, 'grid_scores_'):
        logging.info('No trained model given, building training features for '
                     'binary class codes: %s' % bin(classes))
        model.fit(x_train, y_train)

        logging.info('writing new model to disk at %s..' % model_loc)
        with open(model_loc, 'w') as sink:
            cPickle.dump(model, sink)
        logging.info('done.')

    # get test data
    dev_labels, dev_tweets, dev_pos = preprocessing.parse(dev, classes)
    x_dev = [e[0]+'\t'+e[1] for e in zip(dev_tweets, dev_pos)]
    y_dev = dev_labels
    # x_dev, y_dev = uniformify(x_dev, y_dev)

    # initial eval
    logging.info('Initial evaluation..')
    print_scores(model, x_dev, y_dev, classes)

    # print label distribution, in order to check on how the ratio of pos to neg
    # influences the scoring. Seems pretty balanced now that len(t_gold_1) is
    # equal to len(t_gold_2)
    # print('num t_gold_1 '+str(len([l for l in y_train if l == 1])))
    # print('num t_gold_2 '+str(len([l for l in y_train if l == 2])))
    # print('t_1 '+str(len([l for l in model.predict(x_train) if l == 1])))
    # print('t_2 '+str(len([l for l in model.predict(x_train) if l == 2])))
    # print('num d_gold_1 '+str(len([l for l in y_dev if l == 1])))
    # print('num d_gold_2 '+str(len([l for l in y_dev if l == 2])))
    # print('d_1 '+str(len([l for l in model.predict(x_dev) if l == 1])))
    # print('d_2 '+str(len([l for l in model.predict(x_dev) if l == 2])))

    # feedback loop
    logging.info('Initializing backfeed instance..')
    feed = Feeder(corpus_loc)
    logging.info('done. Now starting backfeed loop')
    for count in range(1, num_runs+1):
        feed.add_best_n(model, num, x_train, y_train)
        logging.info('Retrain run %i' % count)
        print_scores(model, x_dev, y_dev, classes)


def get_best_params(params):
    """
    **Strange**
    Parses a dictionary of 'best_params', as they get returned from a gridsearch.
    :param params: Dictionary of parameter dictionary
    """
    tmp = {}
    for key in params.keys():
        if key not in tmp:
            tmp[key] = {params[key]: 1}
        elif params[key] not in tmp[key]:
            tmp[key][params[key]] = 1
        else:
            tmp[key][params[key]] += 1
    output = '\n'
    for key in tmp.keys():
        output += key+'\n'
        for arg in tmp[key].keys():
            output += '\t%s: %f\n' % (arg, tmp[key][arg]/float(len(params)))
    logging.info(output)


def score(model, model_loc, train, dev, classes):
    """
    Trains and tests the model with the current feature extractors and
    parameters.
    :param model: Model to be tested, either instance of gird_search, or a path
    to a previously saved model
    :param model_loc: If the model was not loaded, a raw one is trained and
    saved to this location
    :param train: File with annotated training tweets
    :param dev: File with annotated development tweets
    :param classes: Amount of classes which get drawn from train and test sets
    :return: best parameters found by the current grid search
    """
    # if model is raw, train it
    if not hasattr(model, 'grid_scores_'):
        logging.info('No trained model given, building training features for '
                     'binary class codes: %s' % bin(classes))
        train_labels, train_tweets, train_pos = preprocessing.parse(
            train, classes
        )
        x_train = [e[0]+'\t'+e[1] for e in zip(train_tweets, train_pos)]
        y_train = train_labels
        # x_train, y_train = uniformify(x_train, y_train)
        # [u'#RonPaul campaign manager Doug Wead : Contrary to media lies , doing JUST GREAT with delegates even after Super Tuesday http://url\t# N N ^ ^ , A P N N , V R A P N R P ^ ^ U']
        model.fit(x_train, y_train)

        logging.info('writing new model to disk at %s..' % model_loc)
        with open(model_loc, 'w') as sink:
            cPickle.dump(model, sink)
        logging.info('done.')

    # get test data
    dev_labels, dev_tweets, dev_pos = preprocessing.parse(dev, classes)
    x_dev = [e[0]+'\t'+e[1] for e in zip(dev_tweets, dev_pos)]
    y_dev = dev_labels
    # x_dev, y_dev = uniformify(x_dev, y_dev)

    # test and evaluate
    print_scores(model, x_dev, y_dev, classes)
    return model.best_params_


def get_perms(seq):
    """
    Returns all possible permutations from all valid subsets of the input
    sequence. This should really be in some kind of utility file.
    """
    subsets = []
    for i in range(len(seq)+1):
        subsets += [''.join(e) for e in combinations(seq, i)]
    perms = set()
    for s in subsets:
        perms |= set([''.join(e) for e in permutations(s)])
    return perms


def main(argv=None):
    """
    Main function call
    See parser description for information
    """
    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    print('start parameters: %s' % args)

    # generate and check label flags
    if args.labels not in get_perms('pneor'):
        logging.error('Label argument not conform, must be a single sequence'
                      'of characters p, n, e, o, and/or r of length 1-5.')
        return 1
    mapping = {'p': preprocessing.POS, 'n': preprocessing.NEG,
               'e': preprocessing.NEU, 'o': preprocessing.OBJ,
               'r': preprocessing.OON}
    classes = 0
    for char in args.labels:
        classes |= mapping[char]

    # generate trained / raw model
    if args.load:
        logging.info('Searching for model at %s, trying load..' % args.model)
        model = cPickle.load(open(args.model))
        logging.info('done.')
    else:
        model = init_model(args.nrc_dir, args.snt_dir)

    # start the right routine
    if args.corpus:
        re_train(model, args.model, args.train, args.dev, args.corpus,
                 args.a_size, args.i_size, classes)
    else:
        params = score(model, args.model, args.train, args.dev, classes)
        get_best_params(params)
    return 0


if __name__ == '__main__':
    sys.exit(main())