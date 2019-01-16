import logging
from collections import namedtuple
import coloredlogs
from Common import *
from chu_liu import Digraph

logging.basicConfig(filename='logger.txt', level=logging.DEBUG)
logger = logging.getLogger()
coloredlogs.install(level='DEBUG')
coloredlogs.install(level='DEBUG', logger=logger)


class Model:

    def __init__(self, parse_data, iteration_number):
        self.all_data = parse_data
        self.iter = iteration_number
        self.w = {}
        self.score = None
        self.feature_extractor = None

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
                    logger.warning("sentence number: {} from {}".format(idx, len(self.all_data)))
                full_graph = build_full_graph(len(sentence))
                get_score_func = self.score(sentence, self.w)
                digraph = Digraph(full_graph, get_score_func)
                graph = build_real_graph(sentence)
                mst = digraph.mst().successors
                if not is_equal(mst, graph):
                    flag = False
                    temp_w = operation(self.w, self.feature_extractor(sentence, graph), '+')
                    self.w = operation(temp_w, self.feature_extractor(sentence, mst), '-')
        logger.critical("workout complete")
        self.save_w()

    def save_w(self):
        pass

    def test(self, test_data):
        words = 0
        correct = 0
        for idx, sentence in enumerate(test_data):
            words += len(sentence)
            new_sentence = self.infer(sentence)
            correct += sum([1 for i in range(len(sentence)) if new_sentence[i].head == sentence[i].head])
        return correct/words

    def infer(self, sentence):
        full_graph = build_full_graph(len(sentence))
        get_score_func = self.score(sentence, self.w)
        digraph = Digraph(full_graph, get_score_func)
        mst = digraph.mst().successors
        mst = inverse_graph(mst)
        Word = namedtuple('Word', 'counter token pos head')
        return [Word(counter=word.counter, token=word.token, pos=word.pos, head=mst[word.counter]) for word in sentence]
