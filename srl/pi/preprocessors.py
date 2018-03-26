import numpy as np

from ..lp import Preprocessor
from ..utils import array


class PIPreprocessor(Preprocessor):
    @staticmethod
    def set_sent_params(corpus, vocab_word_corpus, vocab_word_emb, vocab_label):
        for sent in corpus:
            sent.set_word_ids(vocab_word_corpus=vocab_word_corpus,
                              vocab_word_emb=vocab_word_emb)
            sent.set_mark_ids()
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

            m = np.sum(sent.mark_ids, axis=0)
            if len(sent.mark_ids) == 0:
                m = np.zeros(shape=sent.n_words, dtype='int32')

            samples.append(x + [m])

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
                batches.append(map(lambda b: array(b), zip(*batch)))
                batch = []
                prev_n_words = n_words
            batch.append(sample)
        if batch:
            batches.append(map(lambda b: array(b), zip(*batch)))
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
                x.append([x_corpus])

            x_emb = sent.word_ids_emb
            if x_emb is not None:
                x.append([x_emb])

            batches.append(x)

        return batches
