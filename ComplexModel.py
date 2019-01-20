import pickle

from data_parser import parse
from Model import Model
from os.path import isfile
import sys


def complex_feature_extractor(sentence_idx, parent, child, sentence):
    c_pos = sentence[child - 1].pos
    c_token = sentence[child - 1].token
    if parent == 0:
        p_pos = 'ROOT'
        p_token = 'ROOT'
    else:
        p_pos = sentence[parent - 1].pos
        p_token = sentence[parent - 1].token
    l_p_pos = l_c_pos = 'BEGIN'
    r_p_pos = r_c_pos = 'END'
    if child > 1:
        l_c_pos = sentence[child - 2].pos
    if child < len(sentence):
        r_c_pos = sentence[child].pos
    if parent > 1:
        l_p_pos = sentence[parent-2].pos
    if parent < len(sentence):
        r_p_pos = sentence[parent].pos

    distance = str(abs(parent - child))
    # key = 'f1_{}_{}'.format(p_token, p_pos)
    # key = 'f2_{}'.format(p_token)
    # key = 'f3_{}'.format(p_pos)
    # key = 'f4_{}_{}'.format(c_token, c_pos)
    # key = 'f5_{}'.format(c_token)
    # key = 'f6_{}'.format(c_pos)
    # key = 'f_13_{}_{}'.format(p_pos, c_pos)
    feature = ('f8_{}_{}_{}'.format(p_pos, c_token, c_pos),
               'f7_{}_{}_{}_{}'.format(p_token, p_pos, c_token, c_pos),
               'f9_{}_{}_{}'.format(p_token, c_token, c_pos),
               'f10_{}_{}_{}'.format(p_token, p_pos, c_pos),
               'f11_{}_{}_{}'.format(p_token, p_pos, c_token),
               'f12_{}_{}'.format(p_token, c_token),
               'f14_{}_{}'.format(p_pos, distance),
               'f15_{}_{}_{}'.format(p_pos, c_pos, distance),
               'f16_{}_{}'.format(l_c_pos, p_pos),
               'f17_{}_{}_{}'.format(c_pos, l_c_pos, p_pos),
               'f18_{}_{}_{}'.format(c_pos, r_c_pos, p_pos),
               'f19_{}_{}_{}'.format(l_c_pos, r_c_pos, p_pos),
               'f20_{}_{}_{}_{}'.format(c_pos, p_pos, l_c_pos, r_p_pos),
               'f21_{}_{}_{}'.format(p_pos, l_c_pos, r_p_pos),
               'f22_{}_{}_{}'.format(c_pos, l_c_pos, r_p_pos),
               'f23_{}_{}'.format(l_c_pos, r_p_pos),
               'f24_{}_{}_{}_{}'.format(c_pos, r_c_pos, p_pos, l_p_pos),
               'f25_{}_{}_{}'.format(c_pos, p_pos, l_p_pos),
               'f26_{}_{}_{}'.format(c_pos, r_c_pos, l_p_pos),
               'f27_{}_{}_{}_{}'.format(l_c_pos, r_c_pos, l_p_pos, r_p_pos),
               'f28_{}_{}_{}'.format(r_c_pos, l_p_pos, r_p_pos),
               'f29_{}_{}_{}'.format(l_c_pos, l_p_pos, r_p_pos)
               )
    return feature


class ComplexModel(Model):

    def __init__(self, parse_data, feature_extractor, w=None):
        Model.__init__(self, parse_data, feature_extractor)
        if w:
            self.w = w

    def save_w(self):
        fname = 'w_pickle/w_complex_{}'.format(self.iter)
        with open(fname, 'wb') as handle:
            pickle.dump(self.w, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    all_data = parse('data/train.labeled')
    test_data = parse('data/test.labeled')
    n = int(sys.argv[1])
    if isfile('w_pickle/w_complex_{}'.format(n)):
        if isfile('results/results_for_{}_iterations_complex'.format(n)):
            exit(0)
        w = pickle.load(open('w_pickle/w_complex_{}'.format(n), 'rb'))
        simple_model = ComplexModel(all_data, complex_feature_extractor, w)
    else:
        simple_model = ComplexModel(all_data, complex_feature_extractor)
        simple_model.train(n)
    results = simple_model.test(test_data)
    fname = 'results/results_for_{}_iterations_complex'.format(n)
    with open(fname, 'w') as f:
        f.write(str(results))
