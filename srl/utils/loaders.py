import gzip
import cPickle

import numpy as np
import theano

from vocab import UNK


class Loader(object):
    def __init__(self, argv):
        self.argv = argv

    def load(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def load_data(fn):
        with gzip.open(fn, 'rb') as gf:
            return cPickle.load(gf)

    @staticmethod
    def load_key_value_format(fn):
        data = []
        with open(fn, 'r') as f:
            for line in f:
                key, value = line.rstrip().split()
                data.append((key, int(value)))
        return data


class CoNLL09Loader(Loader):
    def load(self, path, data_size=1000000, file_encoding='utf-8', is_test=False):
        corpus = []
        sent = []
        with open(path) as f:
            for line in f:
                elem = [l.decode(file_encoding) for l in line.rstrip().split()]
                if len(elem) > 0:
                    if is_test:
                        sent.append(elem[:14])
                    else:
                        sent.append(elem)
                else:
                    corpus.append(sent)
                    sent = []
                if len(corpus) >= data_size:
                    break
        return corpus


class EmbeddingLoader(Loader):
    def load(self, path, file_encoding='utf-8'):
        word_list = []
        emb = []
        with open(path) as f:
            for line in f:
                line = line.rstrip().decode(file_encoding).split()
                word_list.append(line[0])
                emb.append(line[1:])
        emb = np.asarray(emb, dtype=theano.config.floatX)

        if UNK not in word_list:
            word_list = [UNK] + word_list
            unk_vector = np.mean(emb, axis=0)
            emb = np.vstack((unk_vector, emb))

        return word_list, emb
