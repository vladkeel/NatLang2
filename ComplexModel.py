import logging
import pickle

import coloredlogs
from data_parser import parse
from Model import Model, global_cache

logging.basicConfig(filename='logger.txt', level=logging.DEBUG)
logger = logging.getLogger()
coloredlogs.install(level='DEBUG')
coloredlogs.install(level='DEBUG', logger=logger)


class ComplexModel(Model):

    def __init__(self, parse_data, iteration_number):
        Model.__init__(self, parse_data, iteration_number)

    @staticmethod
    def feature_extractor(sentence_idx, parent, child, sentence):
        if (sentence_idx, parent, child) in global_cache:
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
        key = 'f7_{}_{}_{}_{}'.format(p_token, p_pos, c_token, c_pos)
        feature[key] = 1
        key = 'f9_{}_{}_{}'.format(p_token, c_token, c_pos)
        feature[key] = 1
        key = 'f11_{}_{}_{}'.format(p_token, p_pos, c_token)
        feature[key] = 1
        key = 'f12_{}_{}'.format(p_token, c_token)
        feature[key] = 1
        if child > 1:
            l_token = sentence[child - 2].token
            l_pos = sentence[child - 2].pos
            key = 'f20_{}_{}_{}'.format(c_pos, p_pos, l_pos)
            feature[key] = 1
            key = 'f22_{}_{}_{}_{}'.format(c_pos, p_pos, l_pos, l_token)
            feature[key] = 1
        elif child < len(sentence):
            r_token = sentence[child].token
            r_pos = sentence[child].pos
            key = 'f21_{}_{}_{}'.format(c_pos, p_pos, r_pos)
            feature[key] = 1
            key = 'f23_{}_{}_{}_{}'.format(c_pos, p_pos, r_pos, r_token)
            feature[key] = 1
        global_cache[(sentence_idx, parent, child)] = feature
        return feature

    def save_w(self):
        fname = 'w_complex_{}'.format(self.iter)
        with open(fname, 'wb') as handle:
            pickle.dump(self.w, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    all_data = parse('data/train.labeled')
    test_data = parse('data/test.labeled')
    for n in [20, 50, 80, 100]:
        complex_model = ComplexModel(all_data, n)
        complex_model.train()
        results = complex_model.test(test_data)
        fname = 'results_for_{}_iterations_complex'.format(n)
        with open(fname, 'w') as f:
            f.write(str(results))
