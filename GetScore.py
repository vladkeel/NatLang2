import numpy as np
import scipy
from Common import Common


class GetScore(Common):

    def __init__(self, sentence, w):
        self.sentence = sentence
        self.w = w

    def __call__(self, *args, **kwargs):
        parent, child = args[0], args[1]
        features = self.feature_extractor(self.sentence, parent, child)
        return self.dot(features, self.w)


class GetScoreComplex(Common):
    def __init__(self, sentence, w):
        self.sentence = sentence
        self.w = w

    def __call__(self, *args, **kwargs):
        parent, child = args[0], args[1]
        features = self.feature_extractor_complex(self.sentence, parent, child)
        return self.dot(features, self.w)




