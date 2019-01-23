import pickle
import colorama
colorama.init()
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
    direction = 'LEFT' if parent > child else 'RIGHT'
    feature = (# 'f1_{}_{}'.format(p_token, p_pos),
               # 'f2_{}'.format(p_token),
               # 'f3_{}'.format(p_pos),
               # 'f4_{}_{}'.format(c_token, c_pos),
               # 'f5_{}'.format(c_token),
               # 'f6_{}'.format(c_pos),
               'f7_{}_{}_{}_{}'.format(p_token, p_pos, c_token, c_pos),
               'f8_{}_{}_{}'.format(p_pos, c_token, c_pos),
               'f9_{}_{}_{}'.format(p_token, c_token, c_pos),
               'f10_{}_{}_{}'.format(c_pos, p_token, p_pos),
               'f11_{}_{}_{}'.format(c_token, p_token, p_pos),
               'f12_{}_{}'.format(p_token, c_token),
               'f13_{}_{}'.format(p_pos, c_pos),
               'f14_{}_{}'.format(p_token, c_pos),
               'f15_{}_{}'.format(p_pos, c_token),
               'f16_{}_{}'.format(p_pos, l_c_pos),
               'f17_{}_{}'.format(p_pos, r_c_pos),
               # 'f18_{}_{}'.format(p_token, l_c_pos),
               'f19_{}_{}'.format(p_token, r_c_pos),
               'f20_{}_{}'.format(c_pos, l_p_pos),
               'f21_{}_{}'.format(c_pos, r_p_pos),
               'f22_{}_{}'.format(c_token, l_p_pos),
               # 'f23_{}_{}'.format(c_token, r_p_pos),
               'f24_{}_{}_{}'.format(c_pos, l_c_pos, p_pos),
               'f25_{}_{}_{}'.format(c_pos, r_c_pos, p_pos),
               'f26_{}_{}_{}'.format(c_token, l_c_pos, p_token),
               'f27_{}_{}_{}'.format(c_token, r_c_pos, p_token),
               'f28_{}_{}_{}'.format(c_pos, l_p_pos, p_pos),
               'f29_{}_{}_{}'.format(c_pos, r_p_pos, p_pos),
               # 'f30_{}_{}_{}'.format(c_token, l_p_pos, p_token),
               'f31_{}_{}_{}'.format(c_token, r_p_pos, p_token),
               'f32_{}_{}_{}'.format(p_pos, l_c_pos, r_c_pos),
               'f33_{}_{}_{}'.format(p_token, l_c_pos, r_c_pos),
               'f34_{}_{}_{}'.format(c_pos, l_p_pos, r_p_pos),
               # 'f35_{}_{}_{}'.format(c_token, l_p_pos, r_p_pos),
               'f36_{}_{}'.format(l_c_pos, r_p_pos),
               'f37_{}_{}'.format(r_c_pos, r_p_pos),
               'f38_{}_{}'.format(l_c_pos, l_p_pos),
               'f39_{}_{}'.format(r_c_pos, l_p_pos),
               'f40_{}_{}_{}'.format(p_pos, l_c_pos, r_p_pos),
               'f41_{}_{}_{}'.format(c_pos, l_c_pos, r_p_pos),
               'f42_{}_{}_{}'.format(p_token, l_c_pos, r_p_pos),
               'f43_{}_{}_{}'.format(c_token, l_c_pos, r_p_pos),
               'f44_{}_{}_{}'.format(p_pos, l_c_pos, l_p_pos),
               'f45_{}_{}_{}'.format(c_pos, l_c_pos, l_p_pos),
               'f46_{}_{}_{}'.format(p_token, l_c_pos, l_p_pos),
               # 'f47_{}_{}_{}'.format(c_token, l_c_pos, l_p_pos),
               'f48_{}_{}_{}'.format(p_pos, r_c_pos, r_p_pos),
               'f49_{}_{}_{}'.format(c_pos, r_c_pos, r_p_pos),
               # 'f50_{}_{}_{}'.format(p_token, r_c_pos, r_p_pos),
               'f51_{}_{}_{}'.format(c_token, r_c_pos, r_p_pos),
               'f52_{}_{}_{}'.format(p_pos, r_c_pos, l_p_pos),
               'f53_{}_{}_{}'.format(c_pos, r_c_pos, l_p_pos),
               'f54_{}_{}_{}'.format(p_token, r_c_pos, l_p_pos),
               'f55_{}_{}_{}'.format(c_token, r_c_pos, l_p_pos),
               'f56_{}_{}_{}'.format(c_pos, p_pos, l_p_pos),
               # 'f57_{}_{}_{}'.format(c_token, p_token, l_p_pos),
               'f58_{}_{}_{}'.format(c_pos, p_token, l_p_pos),
               'f59_{}_{}_{}'.format(c_token, p_pos, l_p_pos),
               'f60_{}_{}_{}'.format(c_pos, p_pos, r_p_pos),
               'f61_{}_{}_{}'.format(c_token, p_token, r_p_pos),
               'f62_{}_{}_{}'.format(c_pos, p_token, r_p_pos),
               # 'f63_{}_{}_{}'.format(c_token, p_pos, r_p_pos),
               'f64_{}_{}_{}'.format(c_pos, p_pos, l_c_pos),
               'f65_{}_{}_{}'.format(c_token, p_token, l_c_pos),
               'f66_{}_{}_{}'.format(c_pos, p_token, l_c_pos),
               'f67_{}_{}_{}'.format(c_token, p_pos, l_c_pos),
               'f68_{}_{}_{}'.format(c_pos, p_pos, r_c_pos),
               'f69_{}_{}_{}'.format(c_token, p_token, r_c_pos),
               'f70_{}_{}_{}'.format(c_pos, p_token, r_c_pos),
               'f71_{}_{}_{}'.format(c_token, p_pos, r_c_pos),
               'f72_{}_{}_{}_{}'.format(c_pos, p_pos, l_c_pos, r_p_pos),
               'f73_{}_{}_{}_{}'.format(c_pos, p_pos, l_c_pos, l_p_pos),
               # 'f74_{}_{}_{}_{}'.format(c_pos, p_pos, r_c_pos, r_p_pos),
               'f75_{}_{}_{}_{}'.format(c_pos, p_pos, r_c_pos, l_p_pos),
               # 'f76_{}_{}_{}'.format(c_pos, r_c_pos, l_p_pos),
               # 'f77_{}_{}_{}'.format(c_pos, l_c_pos, l_p_pos),
               # 'f78_{}_{}_{}'.format(c_pos, r_c_pos, r_p_pos),
               # 'f79_{}_{}_{}'.format(c_pos, l_c_pos, r_p_pos),
               # 'f80_{}_{}_{}'.format(p_pos, r_c_pos, l_p_pos),
               # 'f81_{}_{}_{}'.format(p_pos, r_c_pos, r_p_pos),
               # 'f82_{}_{}_{}'.format(p_pos, l_c_pos, l_p_pos),
               # 'f83_{}_{}_{}'.format(p_pos, l_c_pos, r_p_pos),
               'f84_{}_{}_{}_{}'.format(l_c_pos, r_c_pos, l_p_pos, r_p_pos),
               'f85_{}_{}_{}'.format(r_c_pos, l_p_pos, r_p_pos),
               'f86_{}_{}_{}'.format(l_c_pos, l_p_pos, r_p_pos),
               'f87_{}_{}_{}'.format(r_p_pos, l_c_pos, r_c_pos),
               'f88_{}_{}_{}'.format(l_p_pos, l_c_pos, r_c_pos),
               'f89_{}'.format(direction),
               'f90_{}_{}'.format(p_pos, direction),
               'f91_{}_{}'.format(c_pos, direction),
               'f92_{}_{}'.format(p_token, direction),
               'f93_{}_{}'.format(c_token, direction),
               'f94_{}_{}_{}'.format(p_pos, c_pos, direction),
               # 'f95_{}_{}_{}'.format(p_token, c_token, direction),
               'f96_{}'.format(distance),
               'f97_{}_{}'.format(p_pos, distance),
               'f98_{}_{}'.format(c_pos, distance),
               'f99_{}_{}'.format(p_token, distance),
               'f100_{}_{}'.format(c_token, distance),
               'f101_{}_{}_{}'.format(p_pos, c_pos, distance),
               # 'f102_{}_{}_{}'.format(p_token, c_token, distance)
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
