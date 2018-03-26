from collections import Counter
from copy import deepcopy

import numpy as np

from . import array, convert_str_to_id, Vocab, UNK, UNDER_BAR, Sent


class Preprocessor(object):
    def __init__(self, argv):
        self.argv = argv

    @staticmethod
    def make_sents(corpus, is_prd=True):
        """
        :param corpus: 1D: n_sents, 2D: n_words, 3D: n_columns
        :return 1D: n_sents; elem=Sent
        """
        is_test = True if len(corpus[0][0]) < 15 else False
        return [Sent(sent=sent, is_test=is_test) for sent in corpus]

    @staticmethod
    def make_word_list(corpus, cut_word):
        words = []
        for sent in corpus:
            words += sent.forms
        cnt = Counter(words)
        words = [w for w, c in sorted(cnt.iteritems(), key=lambda x: x[1], reverse=True) if c > cut_word]
        return words

    @staticmethod
    def make_vocab_word(word_list):
        vocab_word = Vocab()
        vocab_word.add_word(UNK)
        for w in word_list:
            vocab_word.add_word(w)
        return vocab_word

    @staticmethod
    def make_vocab_label(corpus, vocab_label_tmp=None, cut_label=0):
        if vocab_label_tmp:
            vocab_label = deepcopy(vocab_label_tmp)
        else:
            vocab_label = Vocab()
            vocab_label.add_word(UNDER_BAR)

        labels = []
        for sent in corpus:
            if sent.has_prds:
                for prop in sent.prd_props:
                    labels += prop
        cnt = Counter(labels)
        labels = [(w, c) for w, c in sorted(cnt.iteritems(), key=lambda x: x[1], reverse=True) if c > cut_label]

        for label, count in labels:
            vocab_label.add_word(label)

        return vocab_label

    @staticmethod
    def set_sent_params(corpus, vocab_word_corpus, vocab_word_emb, vocab_label):
        for sent in corpus:
            sent.set_word_ids(vocab_word_corpus=vocab_word_corpus,
                              vocab_word_emb=vocab_word_emb)
            sent.set_mark_ids()
            if vocab_label:
                sent.set_label_ids(vocab_label=vocab_label)
        return corpus

    @staticmethod
    def make_samples(corpus):
        samples = []
        for sent in corpus:
            x = []

            x_corpus = sent.word_ids_corpus
            if x_corpus is not None:
                x.append(x_corpus)

            x_emb = sent.word_ids_emb
            if x_emb is not None:
                x.append(x_emb)

            samples += map(lambda m, y: x + [m, y], sent.mark_ids, sent.label_ids)

        return samples

    def make_batches(self, samples):
        """
        :param samples: 1D: n_samples, 2D: [x, m, y]
        :return 1D: n_batches, 2D: batch_size; elem=[x, m, y]
        """
        np.random.shuffle(samples)
        samples.sort(key=lambda sample: len(sample[0]))

        batches = []
        batch = []
        prev_n_words = len(samples[0][0])
        for sample in samples:
            n_words = len(sample[0])
            if len(batch) == self.argv.batch_size or prev_n_words != n_words:
                batches.append(map(lambda b: b, zip(*batch)))
                batch = []
                prev_n_words = n_words
            batch.append(sample)
        if batch:
            batches.append(map(lambda b: b, zip(*batch)))
        return batches

    @staticmethod
    def make_test_batches(corpus):
        """
        :param corpus: 1D: n_sents, 2D: n_prds; elem=(x1, x2, y); x1, x2, y: 1D: n_words
        :return 1D: n_batches, 2D: batch_size; elem=(x1, x2, y)
        """
        batches = []
        for sent in corpus:
            x = []

            x_corpus = sent.word_ids_corpus
            if x_corpus is not None:
                x.append(x_corpus)

            x_emb = sent.word_ids_emb
            if x_emb is not None:
                x.append(x_emb)

            batch = map(lambda m: x + [m], sent.mark_ids)
            batches.append(map(lambda b: b, zip(*batch)))

        return batches

    @staticmethod
    def separate_x_and_y_in_batch(batches, index=1):
        """
        :param batches: 1D: n_batches, 2D: batch_size; elem=(x1, ..., y)
        :return 1D: n_batches, 2D: batch_size; elem=(x1, ...)
        :return 1D: n_batches, 2D: batch_size; elem=y
        """
        x = []
        y = []
        for batch in batches:
            x.append(batch[:-index])
            y.append(batch[-index])
        return x, y

    @staticmethod
    def make_sample_from_input_sent(sent, vocab_word_corpus, vocab_word_emb):
        sent = [w.lower() for w in sent]
        x = []
        if vocab_word_corpus:
            word_ids_corpus = array([w for w in convert_str_to_id(sent=sent,
                                                                  vocab=vocab_word_corpus,
                                                                  unk=UNK)])
            x.append([word_ids_corpus])
        if vocab_word_emb:
            word_ids_emb = array([w for w in convert_str_to_id(sent=sent,
                                                               vocab=vocab_word_emb,
                                                               unk=UNK)])
            x.append([word_ids_emb])
        return x

    @staticmethod
    def add_mark_to_sample(sample, mark):
        x = [x_i[0] for x_i in sample]
        prd_indices = [i for i, m in enumerate(mark[0]) if m]

        n_words = len(x[0])
        n_prds = len(prd_indices)

        mark_ids = [[0 for _ in xrange(n_words)] for _ in xrange(n_prds)]
        for i, prd_index in enumerate(prd_indices):
            mark_ids[i][prd_index] = 1
        mark_ids = array(mark_ids)

        batch = map(lambda m: x + [m], mark_ids)
        batch = map(lambda b: b, zip(*batch))
        return batch
