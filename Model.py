from collections import namedtuple, Counter
from Common import *
from chu_liu import Digraph

logger = Logger()
global_cache = {}
global_graph_cache = {}


class GetScore:

    def __init__(self, sentence_idx, w, sentence, feature_extractor):
        self.feature_extractor = feature_extractor
        self.sentence_idx = sentence_idx
        self.sentence = sentence
        self.w = w
        self.cache = {}

    def __call__(self, *args, **kwargs):
        parent, child = args[0], args[1]
        key = (parent, child)
        if key in self.cache:
            return self.cache[key]
        features = self.feature_extractor(self.sentence_idx, parent, child, self.sentence)
        self.cache[key] = dot(self.w, features)
        return self.cache[key]


class Model:

    def __init__(self, parse_data, feature_extractor):
        self.feature_extractor = feature_extractor
        self.all_data = parse_data
        self.iter = 0
        self.w = {}
        self.score = None

    def graph_feature_extractor(self, sentence_idx, graph, sentence):
        features = []
        for key, value in graph.items():
            for vertex in value:
                features.extend(self.feature_extractor(sentence_idx, key, vertex, sentence))
        sum_features = Counter(features)
        return sum_features

    def perceptron(self):
        flag = True
        self.iter += 1
        for idx, sentence in enumerate(self.all_data, start=1):
            full_graph = build_full_graph(len(sentence))
            get_score_func = GetScore(idx, self.w, sentence, self.feature_extractor)
            digraph = Digraph(full_graph, get_score_func)
            graph = build_real_graph(sentence)
            mst = digraph.mst().successors
            add_graph = {}
            rm_graph = {}
            for k in graph.keys():
                add_graph[k] = [v for v in graph[k] if v not in mst[k]]
                rm_graph[k] = [v for v in mst[k] if v not in graph[k]]
            if any(add_graph.values()):
                flag = False
                temp_w = plus(self.w, self.graph_feature_extractor(idx, add_graph, sentence))
                self.w = minus(temp_w, self.graph_feature_extractor(idx, rm_graph, sentence))
            progress_bar(idx, len(self.all_data), "sentences")
        return flag

    def train(self, n):
        logger.critical("Start training")
        for i in range(n):
            logger.debug("iteration {} from {}".format(i+1, self.iter))
            if self.perceptron():
                break
        logger.critical("Training complete")
        self.save_w()

    def save_w(self):
        pass

    def test(self, test_data):
        words = 0
        correct = 0
        logger.critical("Start testing")
        for idx, sentence in enumerate(test_data, start=1):
            words += len(sentence)
            new_sentence = self.infer(sentence)
            correct += sum([1 for i in range(len(sentence)) if new_sentence[i].head == sentence[i].head])
            progress_bar(idx, len(test_data), "sentences")
        logger.critical("Test complete.")
        return correct/words

    def infer(self, sentence):
        full_graph = build_full_graph(len(sentence))
        get_score_func = GetScore(-1, self.w, sentence, self.feature_extractor)
        digraph = Digraph(full_graph, get_score_func)
        mst = digraph.mst().successors
        mst = inverse_graph(mst)
        Word = namedtuple('Word', 'counter token pos head')
        return [Word(counter=word.counter, token=word.token, pos=word.pos, head=mst[word.counter]) for word in sentence]
