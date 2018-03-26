import sys

import numpy as np
import theano


def write(s, stream=sys.stdout):
    stream.write(s + '\n')
    stream.flush()


def convert_str_to_id(sent, vocab, unk):
    """
    :param sent: 1D: n_words
    :param vocab: Vocab()
    :param unk: str
    :return: 1D: n_words; elem=id
    """
    return map(lambda w: vocab.get_id(w) if vocab.has_key(w) else vocab.get_id(unk), sent)


def array(sample, is_float=False):
    if is_float:
        return np.asarray(sample, dtype=theano.config.floatX)
    return np.asarray(sample, dtype='int32')


def average_vector(emb):
    return np.mean(np.asarray(emb[2:], dtype=theano.config.floatX), axis=0)
