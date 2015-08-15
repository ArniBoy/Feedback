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

t0 = time()
num_samples = 5000
reduction = 0.5

count = 0
data = []
for line in codecs.open('/Users/Ceca/Arne/Data/Corpora/batches/tokenized.tsv', encoding='utf-8'):
    data.append(line.strip().split('\t')[0])
    count += 1
    if count > num_samples:
        break

draw = False
print('num samples: %i, feature reduction: %.2f%%' % (num_samples, reduction*100))
print('Draw is %s' % draw)


clf = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
X = clf.fit_transform(data)

print("done in %fs" % (time() - t0))
print("n_features before lsa: %d" % X.shape[1])

svd = TruncatedSVD(int(X.shape[1]*reduction))
lsa = make_pipeline(svd, Normalizer(copy=False))

X = lsa.fit_transform(X)

print("done in %fs" % (time() - t0))
print("n_features after lsa: %d" % X.shape[1])

explained_variance = svd.explained_variance_ratio_.sum()
print("Explained variance of the SVD step: {}%".format(
    int(explained_variance * 100)))

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

# Compute DBSCAN
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

    for label in set(labels):
        labelled_samples = (labels == label) & core_samples_mask
        for idx in [i for i in labelled_samples if i is not False]:
            pass
            #print('%s' % data[idx])



exit()

for k in range(2, 20, 1):
    km = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000)

    print("\n\nClustering sparse data with k=%i" % k)
    km.fit(X)

    print("Top terms per cluster:", end='')
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = clf.get_feature_names()
    for i in range(k):
        print("\nCluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
