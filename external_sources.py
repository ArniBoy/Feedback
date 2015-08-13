__author__ = 'arne'

from codecs import open

from preprocessing import POS, NEU, NEG, OBJ, OON


class SentMutate(object):
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
        :param weight: the amount of influence granted to this procedure
        :return: a real number, to be added on top of current decision_function
        distance values
        """
        raise NotImplementedError("Class %s doesn't implement aMethod()" %
                                  (self.__class__.__name__,))


class AFinnWordList(SentMutate):
    def __init__(self, loc='/home/arne/Masterarbeit/data/afinn/AFINN-111.txt'):
        """
        A mutator which employs the AFINN weighted word list
        (http://arxiv.org/abs/1103.2903)


        :param loc: Location of the weighted word list
        """
        self.pos_idx = -1
        self.neg_idx = -1
        self.ranges = {}
        self.weight = -1
        self.w_l = {}
        for line in open(loc, 'r', 'utf-8'):
            k, v = line.strip().lower().split('\t')
            self.w_l[k] = int(v)

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
        if not self.ranges:
            raise RuntimeError("Tried to run filter without specified ranges!")
        score = 0.0
        for word in tweet.lower().split(' '):
            sent = self.w_l.get(word, None)
            if sent is not None:
                score += sent
        return self.ranges[str(label)][0] < score < self.ranges[str(label)][1]

    def apply_weighting(self, tweet, distances):
        if self.weight < 0 or self.pos_idx < 0 or self.neg_idx < 0:
            raise RuntimeError("Tried to run weighting without specified data!")
        score = 0.0
        words = tweet.lower().split(' ')
        for word in words:
            sent = self.w_l.get(word, None)
            if sent is not None:
                score += sent
        if score > 0:
            distances[self.pos_idx] += self.weight * score / len(words)
        elif score < 0:
            distances[self.neg_idx] -= self.weight * score / len(words)
        return distances
