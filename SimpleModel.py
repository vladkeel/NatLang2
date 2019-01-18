import logging
import pickle
from data_parser import parse
import coloredlogs
from Model import Model, global_cache
from os.path import isfile
import sys

logging.basicConfig(filename='logger.txt', level=logging.DEBUG)
logger = logging.getLogger()
coloredlogs.install(level='DEBUG')
coloredlogs.install(level='DEBUG', logger=logger)


def feature_extractor(sentence_idx, parent, child, sentence):
    if sentence_idx != -1 and (sentence_idx, parent, child) in global_cache:
        return global_cache[(sentence_idx, parent, child)]
    c_pos = sentence[child - 1].pos
    c_token = sentence[child - 1].token
    if parent == 0:
        p_pos = 'ROOT'
        p_token = 'ROOT'
    else:
        p_pos = sentence[parent - 1].pos
        p_token = sentence[parent - 1].token
    feature = {}
    key = 'f1_{}_{}'.format(p_token, p_pos)
    feature[key] = 1
    key = 'f2_{}'.format(p_token)
    feature[key] = 1
    key = 'f3_{}'.format(p_pos)
    feature[key] = 1
    key = 'f4_{}_{}'.format(c_token, c_pos)
    feature[key] = 1
    key = 'f5_{}'.format(c_token)
    feature[key] = 1
    key = 'f6_{}'.format(c_pos)
    feature[key] = 1
    key = 'f8_{}_{}_{}'.format(p_pos, c_token, c_pos)
    feature[key] = 1
    key = 'f10_{}_{}_{}'.format(p_token, p_pos, c_pos)
    feature[key] = 1
    key = 'f_13_{}_{}'.format(p_pos, c_pos)
    feature[key] = 1
    if sentence_idx != -1:
        global_cache[(sentence_idx, parent, child)] = feature
    return feature

class SimpleModel(Model):

    def __init__(self, parse_data, iteration_number, feature_extractor, w=None):
        Model.__init__(self, parse_data, iteration_number, feature_extractor)
        if w:
            self.w = w

    def save_w(self):
        fname = 'w_simple_{}'.format(self.iter)
        with open(fname, 'wb') as handle:
            pickle.dump(self.w, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    all_data = parse('data/train.labeled')
    test_data = parse('data/test.labeled')
    #for n in [20, 50, 80, 100]:
    n = int(sys.argv[1])
    if isfile('w_simple_{}'.format(n)):
        if isfile('results_for_{}_iterations_simple'.format(n)):
            exit(0)
        w = pickle.load(open('w_simple_{}'.format(n), 'rb'))
        print("w:\n{}".format(w))
        simple_model = SimpleModel(all_data, n, feature_extractor, w)
    else:
        simple_model = SimpleModel(all_data, n, feature_extractor)
        simple_model.train()
    results = simple_model.test(test_data)
    fname = 'results_for_{}_iterations_simple'.format(n)
    with open(fname, 'w') as f:
        f.write(str(results))


