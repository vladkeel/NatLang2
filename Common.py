from datetime import datetime
import sys


def progress_bar(part, all_part, word):
    """
    Print progress bar
    :param part: integer - number of iterations completed
    :param all_part: complete number of iterations
    :param word: text to print at the end
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
    return sum([a.get(key, 0) for key in b])


def plus(a, b):
    for k, v in b.items():
        a[k] = a.get(k, 0) + v
    return a


def minus(a, b):
    for k, v in b.items():
        a[k] = a.get(k, 0) - v
    return a


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

if __name__ == '__main__':
    graph = build_full_graph(4)
    print(graph)
