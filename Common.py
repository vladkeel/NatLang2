from collections import defaultdict
from functools import reduce
from datetime import datetime
import sys


def progress_bar(part, all_part, word):
    """
    Prints progress bar to console
    Args:
        progress: float in [0,1] representing progress in action
                    where 0 nothing done and 1 completed.
        text: Short string to add after progress bar.
    """
    progress = float(part)/all_part
    text = "\033[1;36m{}\033[0;32m of \033[1;36m{}\033[0;32m {}.".format(part, all_part, word)
    if isinstance(progress, int):
        progress = float(progress)
    block = int(round(20 * progress))
    progress_line = "\r\033[0;32mCompleted: \033[1;31m[{0}] {1:5.2f}% \033[0;32m{2}.\033[0;0m"\
        .format("#" * block + "-" * (20 - block), progress * 100, text)
    sys.stdout.write(progress_line)
    if progress == 1:
        sys.stdout.write('\n')
    sys.stdout.flush()


class Logger:
    def __init__(self, file=None):
        self.handle = open(file, 'w') if file else sys.stdout

    def warning(self, text):
        sys.stdout.write("\033[0;35m{} \033[0;0m{} \033[1;34m{}\033[0;0m\n".format(datetime.now(), "WARNING", text))

    def critical(self, text):
        sys.stdout.write("\033[0;35m{} \033[0;0m{} \033[1;31m{}\033[0;0m\n".format(datetime.now(), "CRITICAL", text))

    def debug(self, text):
        sys.stdout.write("\033[0;35m{} \033[0;0m{} \033[0;32m{}\033[0;0m\n".format(datetime.now(), "DEBUG", text))

def feature_extractor(sentence, parent, child):
    c_pos = sentence[child-1].pos
    c_token = sentence[child-1].token
    if parent == 0:
        p_pos = 'ROOT'
        p_token = 'ROOT'
    else:
        p_pos = sentence[parent-1].pos
        p_token = sentence[parent-1].token
    feature = {}
    key = 'f1_{}_{}'.format(p_token, p_pos)
    feature[key] = 1
    key = 'f2_{}'.format(p_token)
    feature[key] = 1
    key = 'f3_{}'.format(p_pos)
    feature[key] = 1
    key = 'f4_{}_{}'.format(c_token, c_pos)
    feature[key] = 1
    key = 'f5_{}'.format(c_token)
    feature[key] = 1
    key = 'f6_{}'.format(c_pos)
    feature[key] = 1
    key = 'f8_{}_{}_{}'.format(p_pos, c_token, c_pos)
    feature[key] = 1
    key = 'f10_{}_{}_{}'.format(p_token, p_pos, c_pos)
    feature[key] = 1
    key = 'f_13_{}_{}'.format(p_pos, c_pos)
    feature[key] = 1

    return feature


def feature_extractor_complex(sentence, parent, child):
    feature = feature_extractor(sentence, parent, child)
    c_pos = sentence[child-1].pos
    c_token = sentence[child-1].token
    if parent == 0:
        p_pos = 'ROOT'
        p_token = 'ROOT'
    else:
        p_pos = sentence[parent-1].pos
        p_token = sentence[parent-1].token
    key = 'f7_{}_{}_{}_{}'.format(p_token, p_pos, c_token, c_pos)
    feature[key] = 1
    key = 'f9_{}_{}_{}'.format(p_token, c_token, c_pos)
    feature[key] = 1
    key = 'f11_{}_{}_{}'.format(p_token, p_pos, c_token)
    feature[key] = 1
    key = 'f12_{}_{}'.format(p_token, c_token)
    feature[key] = 1
    if child > 1:
        l_token = sentence[child-2].token
        l_pos = sentence[child-2].pos
        key = 'f20_{}_{}_{}'.format(c_pos, p_pos, l_pos)
        feature[key] = 1
        key = 'f22_{}_{}_{}_{}'.format(c_pos, p_pos, l_pos, l_token)
        feature[key] = 1
    elif child < len(sentence):
        r_token = sentence[child].token
        r_pos = sentence[child].pos
        key = 'f21_{}_{}_{}'.format(c_pos, p_pos, r_pos)
        feature[key] = 1
        key = 'f23_{}_{}_{}_{}'.format(c_pos, p_pos, r_pos, r_token)
        feature[key] = 1

    return feature


def graph_feature_extractor(sentence, graph):
    dicts = []
    for key, value in graph.items():
        for vertex in value:
            dicts.append(feature_extractor(sentence, key, vertex))
    sum_dict = reduce(plus, dicts, {})
    return sum_dict


def graph_feature_extractor_complex(sentence, graph):
    dicts = []
    for key, value in graph.items():
        for vertex in value:
            dicts.append(feature_extractor_complex(sentence, key, vertex))
    sum_dict = reduce(plus, dicts, {})
    return sum_dict


def build_full_graph(size):
    graph = {}
    for vertex in range(size + 1):
        graph[vertex] = [x for x in range(size + 1) if x != vertex and x != 0]
    return graph


def build_real_graph(sentence):
    graph = {0: [w.counter for w in sentence if w.head == 0]}
    for word in sentence:
        graph[word.counter] = [w.counter for w in sentence if w.head == word.counter]
    return graph


def dot(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum([a[key]*b[key] for key in a.keys() if key in b])


def dot_t(a, b):
    if len(a) < len(b):
        return sum([a[key]*b.get(key, 0) for key in a.keys()])
    else:
        return sum([b[key]*a.get(key, 0) for key in b.keys()])


def plus(a,b):
    for k,v in b.items():
        a[k] = a.get(k,0) + v
    return a


def minus(a,b):
    for k,v in b.items():
        a[k] = a.get(k,0) - v
    return a

def operation_t(a, b, op):
    ret = defaultdict(int)
    for k,v in a.items():
        ret[k] += v
    if op == '+':
        for k,v in b.items():
            ret[k] += v
    else:
        for k,v in b.items():
            ret[k] -= v
    return ret


def operation(a, b, op):
    opa = 1 if op == '+' else -1
    ret_val = {}
    for key in a.keys():
        if key in b:
            ret_val[key] = a[key] + (b[key]*opa)
        else:
            ret_val[key] = a[key]
    for key in b.keys():
        if key not in a:
            ret_val[key] = b[key]*opa
    return ret_val


def inverse_graph(graph):
    ret_graph = {}
    for key, values in graph.items():
        for val in values:
            ret_graph[val] = key
    return ret_graph


def is_equal(a, b):
    for key in a.keys():
        if set(a[key]) != set(b[key]):
            return False
    return True
