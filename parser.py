from collections import namedtuple


def parse(fname):
    Word = namedtuple('Word', 'counter token pos head')
    all_data = []
    sentences = []
    with open(fname, 'r') as f:
        sentence = []
        for line in f.readlines():
            if line == '\n':
                sentences.append(sentence)
                sentence = []
                continue
            sentence.append(line.split('\t'))

        for sentence in sentences:
            parsed_sentence = []
            for word in sentence:
                parsed_sentence.append(Word(counter=word[0], token=word[1], pos=word[3], head=word[6]))
            all_data.append(parsed_sentence)
    return all_data
