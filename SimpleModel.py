import logging
import pickle

import coloredlogs
from Common import *
from parser import parse
from Model import Model

logging.basicConfig(filename='logger.txt', level=logging.DEBUG)
logger = logging.getLogger()
coloredlogs.install(level='DEBUG')
coloredlogs.install(level='DEBUG', logger=logger)


class GetScore:

    def __init__(self, sentence, w):
        self.sentence = sentence
        self.w = w

    def __call__(self, *args, **kwargs):
        parent, child = args[0], args[1]
        features = feature_extractor(self.sentence, parent, child)
        return dot(features, self.w)


class SimpleModel(Model):

    def __init__(self, parse_data, iteration_number):
        Model.__init__(self, parse_data, iteration_number)
        self.score = GetScore
        self.feature_extractor = graph_feature_extractor

    def save_w(self):
        fname = 'w_simple_{}'.format(self.iter)
        with open(fname, 'wb') as handle:
            pickle.dump(self.w, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    all_data = parse('data/train.labeled')
    test_data = parse('data/test.labeled')
    for n in [20, 50, 80, 100]:
        simple_model = SimpleModel(all_data, n)
        simple_model.train()
        results = simple_model.test(test_data)
        fname = 'results_for_{}_iterations_simple'.format(n)
        with open(fname, 'w') as f:
            f.write(results)


