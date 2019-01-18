from parser import parse
from os.path import isfile
import pickle
from SimpleModel import SimpleModel

all_data = parse('data/train.labeled')
test_data = parse('data/test.labeled')
for n in [100]:
    if isfile('w_simple_{}'.format(n)):
        if isfile('results_for_{}_iterations_simple'.format(n)):
            continue
        w = pickle.load(open('w_simple_{}'.format(n), 'rb'))
        simple_model = SimpleModel(all_data, n, w)
    else:
        simple_model = SimpleModel(all_data, n)
        simple_model.train()
    results = simple_model.test(test_data)
    fname = 'results_for_{}_iterations_simple'.format(n)
    with open(fname, 'w') as f:
        f.write(str(results))