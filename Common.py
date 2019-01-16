class Common:

    @staticmethod
    def feature_extractor(sentence, parent, child):
        try:
            c_pos = sentence[child-1].pos
        except:
            a=1
        c_token = sentence[child-1].token
        p_pos = p_token = None
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

    def feature_extractor_complex(self, sentence, parent, child):
        pass

    def graph_feature_extractor(self, sentence, graph):
        sum_dict = {}
        for key, value in graph.items():
            for vertex in value:
                sum_dict = self.operation(self.feature_extractor(sentence, key, vertex), sum_dict, '+')
        return sum_dict

    @staticmethod
    def build_full_graph(size):
        graph = {}
        for vertex in range(size + 1):
            graph[vertex] = [x for x in range(size + 1) if x != vertex and x != 0]
        return graph

    @staticmethod
    def build_real_graph(sentence):
        graph = {0: [w.counter for w in sentence if w.head == 0]}
        for word in sentence:
            graph[word.counter] = [w.counter for w in sentence if w.head == word.counter]
        return graph

    @staticmethod
    def dot(a, b):
        return sum([a.get(key, 0)*b.get(key, 0) for key in a.keys()])

    @staticmethod
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

    @staticmethod
    def inverse_graph(graph):
        ret_graph = {}
        for key, values in graph.items():
            for val in values:
                ret_graph[val] = key
        return ret_graph

    @staticmethod
    def is_equal(a, b):
        for key in a.keys():
            if set(a[key]) != set(b[key]):
                return False
        return True



