class Common:

    def feature_extractor(self, sentence, parent, child):
        pass

    def feature_extractor_complex(self, sentence, parent, child):
        pass

    @staticmethod
    def dot(a, b):
        return sum([a.get(key, 0)*b.get(key, 0) for key in a.keys()])
