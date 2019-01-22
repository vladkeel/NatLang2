from data_parser import parse
from ComplexModel import ComplexModel, complex_feature_extractor_tt
import colorama
colorama.init()

if __name__ == '__main__':
    n = 5
    all_data = parse('data/train.labeled')
    test_data = parse('data/test.labeled')
    fname = 'results/compare_features_master'
    with open(fname, 'w') as f:
        f.write('Feat#\tTest acc\n')
        for i in range(89, 103):
            global_test = i
            print("{}! Hoora!".format(i))
            model = ComplexModel(all_data, complex_feature_extractor_tt, special_feature=i)
            model.train(n)
            res = model.test(test_data)
            f.write('{0}\t{1:8.5f}\n'.format(global_test, res))
