from Model import logger
from data_parser import parse
import sys
from SimpleModel import SimpleModel, simple_feature_extractor
from ComplexModel import ComplexModel, complex_feature_extractor


if __name__ == '__main__':
    n = int(sys.argv[2])
    t = sys.argv[1]
    all_data = parse('data/train.labeled')
    test_data = parse('data/test.labeled')
    if t == '-c':
        model_type = 'complex'
        model = ComplexModel(all_data, complex_feature_extractor)
    else:
        model_type = 'simple'
        model = SimpleModel(all_data, simple_feature_extractor)
    fname = 'results/compare_{}_iterations_{}'.format(n, model_type)
    with open(fname, 'w') as f:
        logger.critical("Start training")
        f.write('It#\tTest acc\t Train acc\n')
        for i in range(n):
            logger.debug("iteration {} from {}".format(i + 1, n))
            if model.perceptron():
                break
            result = model.test(test_data)
            result2 = model.test(all_data[:1000])
            f.write('{}\t{}.6f\t{}.6f'.format(i+1, result, result2))
        model.save_w()
        logger.critical("Training complete")
