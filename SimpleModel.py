import logging
from collections import namedtuple

import coloredlogs

from Common import Common
from GetScore import GetScore
from chu_liu import Digraph
from parser import parse

logging.basicConfig(filename='logger.txt', level=logging.DEBUG)
logger = logging.getLogger()
coloredlogs.install(level='DEBUG')
coloredlogs.install(level='DEBUG', logger=logger)


class SimpleModel(Common):

    def __init__(self, parse_data, iteration_number):
        self.all_data = parse_data
        self.iter = iteration_number
        self.w = {}

    def train(self):
        logger.critical("starting training")
        flag = False
        for i in range(self.iter):
            logger.debug("iteration number: {}".format(i+1))
            if flag:
                break
            flag = True
            for idx, sentence in enumerate(self.all_data, start=1):
                if idx % 1000 == 0:
                    logger.debug("sentence number: {} from {}".format(idx, len(self.all_data)))
                full_graph = self.build_full_graph(len(sentence))
                get_score_func = GetScore(sentence, self.w)
                digraph = Digraph(full_graph, get_score_func)
                graph = self.build_real_graph(sentence)
                mst = digraph.mst().successors
                if not self.is_equal(mst, graph):
                    flag = False
                    temp_w = self.operation(self.w, self.graph_feature_extractor(sentence, graph), '+')
                    self.w = self.operation(temp_w, self.graph_feature_extractor(sentence, mst), '-')
        logger.critical("workout complete")

    def infer(self, sentence):
        full_graph = self.build_full_graph(len(sentence))
        get_score_func = GetScore(sentence, self.w)
        digraph = Digraph(full_graph, get_score_func)
        mst = digraph.mst().successors
        mst = self.inverse_graph(mst)
        Word = namedtuple('Word', 'counter token pos head')
        return [Word(counter=word.counter, token=word.token, pos=word.pos, head=mst[word.counter]) for word in sentence]


if __name__ == '__main__':
    all_data = parse('data/train.labeled')
    simple_model = SimpleModel(all_data, 20)
    simple_model.train()
    test = parse('data/test.labeled')
    for idx, sentence in enumerate(test):
        new_sentence = simple_model.infer(sentence)
        sent_value = sum([1 for i in range(len(sentence)) if new_sentence[i].head == sentence[i].head])
        sent_accuracy = sent_value/len(sentence)
        logger.debug("sentence number: {} with accuracy:{}".format(idx, sent_accuracy))











