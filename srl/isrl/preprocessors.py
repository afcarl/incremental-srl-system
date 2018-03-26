import numpy as np

from . import Preprocessor, Sent


class ISRLPreprocessor(Preprocessor):
    @staticmethod
    def make_sents(corpus, marked_prd=True):
        """
        :param corpus: 1D: n_sents, 2D: n_words, 3D: n_columns
        :param marked_prd: whether or not the corpus has annotations of target predicates
        :return 1D: n_sents; elem=Sent
        """
        is_test = True if len(corpus[0][0]) < 14 else False
        return [Sent(sent=sent, is_test=is_test, marked_prd=marked_prd) for sent in corpus]

    @staticmethod
    def set_sent_params(corpus, vocab_word_corpus, vocab_word_emb, vocab_label, vocab_lemma=None):
        for sent in corpus:
            sent.set_word_ids(vocab_word_corpus=vocab_word_corpus,
                              vocab_word_emb=vocab_word_emb)
            sent.set_mark_ids()
            if vocab_lemma:
                sent.set_lemma_ids(vocab_lemma=vocab_lemma)
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
                batches.append(self._make_one_batch(batch))
                batch = []
                prev_n_words = n_words
            batch.append(sample)
        if batch:
            batches.append(self._make_one_batch(batch))
        return batches

    def _make_one_batch(self, batch):
        batch_size = len(batch)
        n_words = len(batch[0][0])
        batch = map(lambda b: b, zip(*batch))

        x = [np.asarray(b, dtype='int32') for b in batch[:-2]]
        y_shift = np.asarray(batch[-2], dtype='int32')
        y_label = np.reshape(np.asarray(batch[-1], dtype='int32'),
                             newshape=(batch_size * n_words, -1))
        y_label = self._make_time_step_labels(y_shift, y_label)

        batch = x + [y_shift, y_label]
        return batch

    @staticmethod
    def _make_time_step_labels(y_shift, y_label):
        """
        :param y_shift: 1D: batch_size, 2D: n_words; elem=0/1
        :param y_label: 1D: batch_size * n_words, 2D: n_words; elem=label id
        :return: 1D: n_words, 2D: batch_size * n_words, 3D: n_words; elem=label id
        """
        batch_size_with_prds = y_label.shape[0]
        n_words = y_label.shape[1]
        zeros = np.zeros(shape=(n_words, batch_size_with_prds, n_words), dtype='int32')
        mask = np.zeros(shape=y_shift.shape, dtype='int32')
        for i in xrange(n_words):
            mask[:, i] = y_shift[:, i]
            zeros[i, :, :i+1] += y_label[:, :i+1] * np.reshape(mask, (batch_size_with_prds, 1))
        return zeros

    @staticmethod
    def remove_pad_labels(y_shift, y_label):
        """
        :param y_shift: 1D: batch_size * n_words; elem=0/1
        :param y_label: 1D: batch_size * n_words(prd) * n_words(arg); elem=label id
        :return: 1D: n_prds, 2D: n_words; elem=label id
        """
        y_label_removed_pad = []
        batch_size = len(y_shift)
        n_words = len(y_label) / batch_size
        y_label = np.reshape(y_label, (batch_size, n_words))
        assert len(y_shift) == len(y_label)
        for i, is_prd in enumerate(y_shift):
            if is_prd > 0:
                y_label_removed_pad.append(y_label[i])
        return y_label_removed_pad

    @staticmethod
    def make_test_batches(corpus):
        samples = []
        for sent in corpus:
            x = []

            x_corpus = sent.word_ids_corpus
            if x_corpus is not None:
                x.append([x_corpus])

            x_emb = sent.word_ids_emb
            if x_emb is not None:
                x.append([x_emb])

            y_shift = [np.sum(sent.mark_ids, axis=0)]

            samples.append(x + [y_shift])

        return samples

    @staticmethod
    def make_sample_from_input_word(word, vocab_word_corpus, vocab_word_emb, time_step):
        x = []
        word_id_corpus = vocab_word_corpus.get_id(word) if vocab_word_corpus else None
        if word_id_corpus:
            x.append([word_id_corpus])
        word_id_emb = vocab_word_emb.get_id(word) if vocab_word_emb else None
        if word_id_emb:
            x.append([word_id_emb])
        x.append(time_step)
        return x
