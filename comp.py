from ComplexModel import ComplexModel, complex_feature_extractor
from SimpleModel import SimpleModel, simple_feature_extractor
from data_parser import parse
import pickle


# Simple model comp
with open('data/comp_m1_302575287.wtag', 'w') as f:
    comp_data = parse('data/comp.unlabeled')
    w = pickle.load(open('w_pickle/w_simple_100', 'rb'))
    simple_model = SimpleModel(comp_data, simple_feature_extractor, w)
    for sentence in comp_data:
        result = simple_model.infer(sentence)
        for word in result:
            f.write("{}\t{}\t_\t{}\t_\t_\t{}\t_\t_\t_\n".format(word.counter, word.token, word.pos, word.head))
        f.write("\n")

# Complex model comp
with open('data/comp_m2_302575287.wtag', 'w') as f:
    comp_data = parse('data/comp.unlabeled')
    w = pickle.load(open('w_pickle/w_complex_50', 'rb'))
    complex_model = ComplexModel(comp_data, complex_feature_extractor, w)
    for sentence in comp_data:
        result = complex_model.infer(sentence)
        for word in result:
            f.write("{}\t{}\t_\t{}\t_\t_\t{}\t_\t_\t_\n".format(word.counter, word.token, word.pos, word.head))
        f.write("\n")
