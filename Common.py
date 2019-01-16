
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
    sum_dict = {}
    for key, value in graph.items():
        for vertex in value:
            sum_dict = operation(feature_extractor(sentence, key, vertex), sum_dict, '+')
    return sum_dict


def graph_feature_extractor_complex(sentence, graph):
    sum_dict = {}
    for key, value in graph.items():
        for vertex in value:
            sum_dict = operation(feature_extractor_complex(sentence, key, vertex), sum_dict, '+')
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
    return sum([a.get(key, 0)*b.get(key, 0) for key in a.keys()])


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
