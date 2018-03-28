import numpy as np

from ..utils import Sent
from ..preprocessors import Preprocessor


class ISRLPreprocessor(Preprocessor):
    @staticmethod
    def make_sents(corpus, marked_prd=True):
        """
        :param corpus: 1D: n_sents, 2D: n_words, 3D: n_columns
        :param marked_prd: whether or not the corpus has annotations of target predicates
        :return 1D: n_sents; elem=Sent
        """
        return [Sent(sent=sent, is_test=False, marked_prd=marked_prd) for sent in corpus]

    @staticmethod
    def set_sent_params(corpus, vocab_word_corpus, vocab_word_emb, vocab_label):
        for sent in corpus:
            sent.set_word_ids(vocab_word_corpus=vocab_word_corpus,
                              vocab_word_emb=vocab_word_emb)
            sent.set_mark_ids()
            if vocab_label:
                sent.set_label_id_matrix(vocab_label=vocab_label)
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

            y_shift = np.sum(sent.mark_ids, axis=0)
            y_label = sent.label_id_matrix

            if sent.n_words < 2 or np.sum(y_shift) < 1:
                continue
            samples.append(x + [y_shift, y_label])

        return samples

    @staticmethod
    def make_sample_from_input_word(word, vocab_word_corpus, vocab_word_emb, time_step):
        x = []

        word_id_corpus = vocab_word_corpus.get_id_or_unk_id(word) if vocab_word_corpus else None
        if word_id_corpus is not None:
            x.append(word_id_corpus)

        word_id_emb = vocab_word_emb.get_id_or_unk_id(word) if vocab_word_emb else None
        if word_id_emb is not None:
            x.append(word_id_emb)

        return [[x], time_step]
