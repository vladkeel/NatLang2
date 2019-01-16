import logging
import coloredlogs
from Common import *
from parser import parse
from Model import Model

logging.basicConfig(filename='logger.txt', level=logging.DEBUG)
logger = logging.getLogger()
coloredlogs.install(level='DEBUG')
coloredlogs.install(level='DEBUG', logger=logger)


class GetScoreComplex:
    def __init__(self, sentence, w):
        self.sentence = sentence
        self.w = w

    def __call__(self, *args, **kwargs):
        parent, child = args[0], args[1]
        features = feature_extractor_complex(self.sentence, parent, child)
        return dot(features, self.w)


class ComplexModel(Model):

    def __init__(self, parse_data, iteration_number):
        Model.__init__(self, parse_data, iteration_number)
        self.score = GetScoreComplex
        self.feature_extractor = graph_feature_extractor_complex


if __name__ == '__main__':
    all_data = parse('data/train.labeled')
    simple_model = ComplexModel(all_data, 20)
    simple_model.train()
    test = parse('data/test.labeled')
    for idx, sentence in enumerate(test):
        new_sentence = simple_model.infer(sentence)
        sent_value = sum([1 for i in range(len(sentence)) if new_sentence[i].head == sentence[i].head])
        sent_accuracy = sent_value/len(sentence)
        logger.debug("sentence number: {} with accuracy:{}".format(idx, sent_accuracy))
