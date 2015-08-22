from __future__ import print_function
__author__ = 'Arne'

from time import time
import codecs
from math import sqrt
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics
from sklearn.pipeline import Pipeline, FeatureUnion

from feature_extractors import AllCapsFeatures, HashtagFeatures, \
    PunctuationFeatures, ElongatedFeatures, EmoticonFeatures, SENTFeatures, \
    NRCFeatures, ItemSelector, DataSeparator, PosFeatures

from toy_classifier import init_model
from preprocessing import parse, POS, NEG, NEU

root = '/Users/Ceca/Arne/Data'


def all_features_kmeans():
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
    return Pipeline([
        ('separate', DataSeparator()),
        ('features', features),
        ('kmeans', MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1,
                                   init_size=1000, batch_size=1000))
    ])


def get_train_data():
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
    semeval_train = [e[0]+'\t'+e[1] for e in zip(train_tweets, train_pos)], train_labels
    semeval_dev = [e[0]+'\t'+e[1] for e in zip(dev_tweets, dev_pos)], dev_labels
    semeval_test = [e[0]+'\t'+e[1] for e in zip(test_tweets, test_pos)], test_labels
    return semeval_train, semeval_dev, semeval_test


t0 = time()
num_samples = 10000
reduction = 0.5

count = 0
tweets = []
all_data = []
for line in codecs.open(root+'/Corpora/batches/tokenized.tsv', encoding='utf-8'):
    if count > num_samples:
        break
    tweet, po = line.strip().split('\t')
    tweets.append(tweet)
    all_data.append(('%s\t%s' % (tweet, po)))
    count += 1
print('len all data: %i' % len(all_data))

draw = False
lsa = False
dbscan = False
kmeans = True
print('num samples: %i, feature reduction: %.2f%%' % (num_samples, reduction*100))
print('Draw is %s' % draw)
print('LSA is %s' % lsa)
print('DBSCAN is %s' % dbscan)
print('KMeans is %s' % kmeans)

clf = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
X = clf.fit_transform(tweets)
# TODO include sentiment lexicon features

print("done in %fs" % (time() - t0))
print("n_features before lsa: %d" % X.shape[1])

# reduce feature space. cannot reduce very large corpora with very large feature spaces.
if lsa:
    svd = TruncatedSVD(int(X.shape[1]*reduction))
    lsa_reduction = make_pipeline(svd, Normalizer(copy=False))

    X = lsa_reduction.fit_transform(X)

    print("done in %fs" % (time() - t0))
    print("n_features after lsa: %d" % X.shape[1])

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

# draws the distance matrix of the feature space
if draw:
    buckets = {}
    distances = metrics.pairwise.pairwise_distances(X, metric='manhattan')
    print('distances computed')
    for idx_a, row in enumerate(distances):
        for idx_b, val in enumerate(row):
            if idx_a != idx_b:
                rounded = '%.2f' % val
                if rounded not in buckets:
                    buckets[rounded] = 1
                else:
                    buckets[rounded] += 1

    max_length = 160.0
    max_count = max(buckets.values())
    for key in sorted(buckets.keys()):
        num_bar = int(buckets[key] * max_length / max_count)
        print('%s: %s' % (key, num_bar * '-'))
    print("done in %fs" % (time() - t0))
    exit()

# Compute DBSCAN clusters, does take too long for many instances
if dbscan:
    minimum_per_cluster = 1.0 / 4
    samples_per_cluster = int(X.shape[0] / sqrt(X.shape[0]/2.0) * minimum_per_cluster)
    print('Estimated minimum amount of samples per cluster: %i' % samples_per_cluster)
    eps = (0.3, 0.6, 0.9, 1.10, 1.42, 1.74, 2.01, 2.24, 2.45, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 3.75, 4.00,
           4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 6.75, 7.00, 7.25, 7.50, 7.75, 8.00)
    for elem in eps:
        db = DBSCAN(eps=elem, min_samples=minimum_per_cluster, metric='cosine', algorithm='brute').fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        print('Estimated number of clusters for esp=%.3f: %d' % (elem, n_clusters_))

        # TODO print representing words of each cluster
        for label in set(labels):
            labelled_samples = (labels == label) & core_samples_mask
            for idx in [i for i in labelled_samples if i is not False]:
                pass


def label_counter(cluster_collection):
    total_length = float(sum([len(data) for data in zip(*cluster_collection)[1]]))
    print('total_length: %f' % total_length)
    consistency = 0
    for c, data in cluster_collection:
        print('cluster: %s content length: %s' % (c, len(data)))
        y_pred = model.predict(data)
        label_counts = {}
        for p in y_pred:
            if p in label_counts:
                label_counts[p] += 1
            else:
                label_counts[p] = 1
        consistency += max(label_counts.values()) / total_length
    return consistency


if kmeans:
    model = init_model()
    train, dev, test = get_train_data()
    model.fit(train[0], train[1])
    max_consistency = label_counter([(0, all_data)])
    print('baseline consistency: %f' % max_consistency)

    max_clusters = ()
    km = all_features_kmeans()
    for run in range(50):
        print('run number %i' % run)
        k = 8
        # km = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1,
        #                      init_size=1000, batch_size=1000)
        km.fit(X)

        cluster = {idx: [] for idx in range(k)}
        for entry, c in zip(all_data, km.named_steps['kmeans'].labels_):
            cluster[c].append(entry)
        local_consistency = label_counter(cluster.items())

        if local_consistency > max_consistency:
            max_consistency = local_consistency
            max_clusters = km.named_steps['kmeans'].labels_
            print('max consistency updated to %f' % max_consistency)

    print('most consistent score: %f' % max_consistency)
    with open(root+'/logs/cluster.txt', 'w') as log:
        for line in max_clusters:
            log.write('%s\n' % line)

        # print("Top terms per cluster:", end='')
        # order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        # terms = clf.get_feature_names()
        # for i in range(k):
        #     print("\nCluster %d:" % i, end='')
        #     for ind in order_centroids[i, :10]:
        #         print(' %s' % terms[ind], end='')


# TODO train supervised classifier, run k-means and always keep the model in tmp that had the best cluster fits
