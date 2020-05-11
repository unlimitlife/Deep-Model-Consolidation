import numpy.random


class curriculums(object):
    @staticmethod
    def basic(num_classes):
        return [x for x in range(num_classes)]

    @staticmethod
    def rand(num_classes):
        return list(numpy.random.permutation(num_classes))
