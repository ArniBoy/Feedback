__author__ = 'arne'

from codecs import open

from sklearn.cluster import KMeans

from preprocessing import POS, NEU, NEG, OBJ, OON

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
        raise NotImplementedError("Class %s doesn't implement aMethod()" %
                                  (self.__class__.__name__,))

    def apply_weighting(self, tweet, distances):
        """
        Interface for tweet weighting by sentiment weights
        :param tweet: input tweet, is converted to lowercase for weighting
        application
        :param distances: original distance measures for the tweet
        :return: a real number, to be added on top of current decision_function
        distance values
        """
        raise NotImplementedError("Class %s doesn't implement aMethod()" %
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
        return distances


class AutoCluster(SentMutate):
    def __init__(self, loc):
        """
        A mutator using automatically created word clusters as basis for re-weighting

        :param loc: Location of the clusters
        """
        super(AutoCluster, self).__init__()
        self.clusters = KMeans()
        data = []
        for tweet, label in zip(open(root+'/Corpora/batches/tokenized.tsv', 'r,', 'utf-8'), open(root+loc)):
            data.append(tweet.strip(), )
        self.clusters.fit(*zip(*[(entry.split('\t')[0], entry.strip.split('\t')[1])
                                 for entry in open(loc, 'r', 'utf-8')]))
        for line in open(loc, 'r', 'utf-8'):
            pass

    def get_score(self, tweet):
        pass

    def apply_filter(self, tweet, label):
        if not self.ranges:
            raise RuntimeError("Tried to run filter without specified ranges!")
        score = 0.0

    def apply_weighting(self, tweet, distances):
        if not self.ranges:
            raise RuntimeError("Tried to run weighting without specified data!")
        score = 0.0



