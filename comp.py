from ComplexModel import ComplexModel, complex_feature_extractor
from data_parser import parse
import pickle

with open('data/comp.labeled', 'w') as f:
    comp_data = parse('data/comp.unlabeled')
    w = pickle.load(open('w_pickle/w_complex_20', 'rb'))
    complex_model = ComplexModel(comp_data, complex_feature_extractor, w)
    for sentence in comp_data:
        result = complex_model.infer(sentence)
        for word in result:
            f.write("{}\t{}\t_\t{}\t_\t_\t{}\t_\t_\t_\n".format(word.counter, word.token, word.pos, word.head))
        f.write("\n")
