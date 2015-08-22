from __future__ import print_function
__author__ = 'arne'

from codecs import open
from datetime import datetime

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV

from preprocessing import parse, POS, NEG, NEU
from twitter_analysis import f1
from feature_extractors import AllCapsFeatures, HashtagFeatures, \
    PunctuationFeatures, ElongatedFeatures, EmoticonFeatures, SENTFeatures, \
    NRCFeatures, ItemSelector, DataSeparator, PosFeatures
from backfeed import Feeder
from external_sources import AFinnWordList

root = '/Users/Ceca/Arne/Data'


def f_range(x, y, jump):
    while x < y:
        yield x
        x += jump


def get_semeval_data(text, meta):
    tweets = {}
    for line_1, line_2 in zip(open(text, 'r', 'utf-8'),
                              open(meta, 'r', 'utf-8')):
        tweet, pos, _, _ = line_1.strip().split('\t')
        _, _, sent = line_2.strip().split('\t')
        tweets['\t'.join([tweet, pos])] = sent
    return tweets.keys(), tweets.values()


def get_semeval13_train():
    text = root+'/semeval2013/SemEval2013-task2-train_dev' \
           '/original_combined-shuffle'
    meta = root+'/semeval2013/SemEval2013-task2-train_dev' \
           '/meta'
    tweets = {}
    for line_1, line_2 in zip(open(text), open(meta)):
        tweet, pos, _, _ = line_1.strip().split('\t')
        _, _, sent = line_2.strip().split('\t')
        tweets[u'\t'.join([tweet, pos])] = sent
    return tweets.items()


def get_semeval13_test():
    text = root+'/semeval2013/SemEval2013-task2-test' \
           '/original_combined'
    meta = root+'/semeval2013/SemEval2013-task2-test' \
           '/meta'
    tweets = {}
    for line_1, line_2 in zip(open(text), open(meta)):
        tweet, pos, _, _ = line_1.strip().split('\t')
        _, _, sent = line_2.strip().split('\t')
        print(tweet, pos, sent)
        tweets[u'\t'.join([tweet, pos])] = sent
    return tweets.items()


def get_semeval14_test():
    tweets = {}
    for line in open(root+'/semeval2014/'
                     'SemEval2014-task9-test-twitter2014/original_combined'):
        data = line.strip().split('\t')
        tweets[data[3]] = data[2]
    return tweets.items()


def get_semeval15_test():
    tweets = {}
    for line in open(root+'/semeval2015/'
                     'SemEval2015-task10-test/original_combined'):
        data = line.strip().split('\t')
        tweets[data[3]] = data[2]
    return tweets.items()


def init_model():
    features = FeatureUnion(
        transformer_list=[
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
                ('lex', SENTFeatures(root+'/Sentiment140-Lexicon-v0.1')),
            ])),
            ('nrc', Pipeline([
                ('selector', ItemSelector(key='tweets')),
                ('lex', NRCFeatures(root+'/NRC-Hashtag-Sentiment-Lexicon-v0.1')),
            ])),
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
            'nrc': 0.05
        }
    )
    pipeline = Pipeline([
        ('separate', DataSeparator()),
        ('features', features),
        ('svm', LinearSVC())
    ])
    parameters = {
        'features__word_ngram__tfidf__analyzer': ('word',),
        'features__word_ngram__tfidf__max_df': (0.75,),
        'features__word_ngram__tfidf__ngram_range': ((1, 5),),
        'features__char_ngram__tfidf__analyzer': ('char',),
        'features__char_ngram__tfidf__max_df': (0.75,),
        'features__char_ngram__tfidf__ngram_range': ((1, 5),),
        'svm__C': (0.5,)
        # 'svm__C': (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2,
        #            3, 4, 5, 6, 7, 8, 9, 10)
    }
    return GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)


def run(x_train, y_train, x_test, y_test):
    model = init_model()
    t = datetime.now()
    with open(root+'/logs/%i-%i-%i_%i-%i-%i.log' %
              (t.year, t.month, t.day, t.hour, t.minute, t.second),
              'w') as log:
        log.write('. Positive Negative Neutral\n')
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        pos_f1 = f1(predicted, y_test, POS)
        neg_f1 = f1(predicted, y_test, NEU)
        neu_f1 = f1(predicted, y_test, NEG)
        macro = 1.0/2.0*(pos_f1+neg_f1)
        log.write('0 %s %s %s %s\n' % (pos_f1, neg_f1, neu_f1, macro))
        print('initial')
        print('\tpositive: %f' % pos_f1)
        print('\tneutral: %f' % neg_f1)
        print('\tnegative: %f' % neu_f1)

        # retrain routine
        feed = Feeder(root+'/Corpora/batches/tokenized.tsv')
        af_wl = AFinnWordList(root+'/afinn/AFINN-111.txt')
        af_wl.add_filter_ranges(**{str(POS): (2, float('inf')),
                                   str(NEG): (float('-inf'), -2),
                                   str(NEU): (-2, 2)})
        af_wl.add_weight(5, model.best_estimator_.named_steps['svm'].classes_)
        feed.add_mutator(af_wl)

        for i in range(30):
            print('count nr. %i' % i)
            feed.add_best_n(model, 300, x_train, y_train, False, ['weight'])

            predicted = model.predict(x_test)
            pos_f1 = f1(predicted, y_test, POS)
            neg_f1 = f1(predicted, y_test, NEU)
            neu_f1 = f1(predicted, y_test, NEG)
            macro = 1.0/2.0*(pos_f1+neg_f1)
            log.write('%i %s %s %s %s\n' % (i+1, pos_f1, neg_f1, neu_f1, macro))
            print('\tpositive: %f' % pos_f1)
            print('\tneutral: %f' % neg_f1)
            print('\tnegative: %f' % neu_f1)


def main():
    # load data
    classes = POS | NEU | NEG
    train_loc = root+'/twitterData/train_alternative.tsv'
    dev_loc = root+'/twitterData/dev_alternative.tsv'
    test_loc = root+'/twitterData/test_alternative.tsv'
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

    # run main routine
    run(train[0], train[1], dev[0], dev[1])


main()
