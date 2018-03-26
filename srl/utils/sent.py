from misc import array, convert_str_to_id
from vocab import UNDER_BAR, UNK, VERB
from word import Word


class Sent(object):
    def __init__(self, sent, is_test=True, marked_prd=True):
        self.words = self._make_words(sent=sent, is_test=is_test, marked_prd=marked_prd)

        self.forms = [word.form for word in self.words]
        self.lemmas = [word.lemma for word in self.words]
        self.marks = [word.mark for word in self.words]
        self.senses = [word.sense for word in self.words]
        self.props = [word.prop for word in self.words]

        self.prd_indices = self._set_prd_indices(self.marks, sent)
        self.prd_forms = [self.forms[i] for i in self.prd_indices]
        self.prd_props = self._set_prd_props(self.props, self.marks, self.prd_indices, is_test)
        self.has_prds = True if len(self.prd_indices) > 0 else False

        self.n_words = len(sent)
        self.n_prds = len(self.prd_indices)

        self.word_ids_corpus = None
        self.word_ids_emb = None
        self.mark_ids = None
        self.label_ids = None
        self.label_id_matrix = None

        self.results = None

    def _make_words(self, sent, is_test, marked_prd=True):
        return [self._make_word(line, is_test, marked_prd) for line in sent]

    @staticmethod
    def _make_word(line, is_test=False, marked_prd=True):
        return Word(form=line[1].lower(),
                    lemma=line[3],
                    mark=line[12] if marked_prd else UNDER_BAR,
                    sense=line[13] if is_test is False else None,
                    prop=line[14:] if is_test is False else [])

    @staticmethod
    def _set_prd_indices(marks, sent):
        poss = [w[4] for w in sent]
        return [i for i in xrange(len(marks)) if marks[i] == 'Y' and poss[i].startswith(VERB)]

    @staticmethod
    def _set_prd_props(props, marks, prd_indices, is_test=True):
        if is_test:
            return []
        props = [prop for prop in map(lambda p: p, zip(*props))]
        all_prd_indices = [i for i in xrange(len(marks)) if marks[i] == 'Y']
        indices = [all_prd_indices.index(p) for p in prd_indices]
        return [props[i] for i in indices]

    def set_word_ids(self, vocab_word_corpus, vocab_word_emb):
        if vocab_word_corpus:
            self.word_ids_corpus = array([w for w in convert_str_to_id(sent=self.forms,
                                                                       vocab=vocab_word_corpus,
                                                                       unk=UNK)])
        if vocab_word_emb:
            self.word_ids_emb = array([w for w in convert_str_to_id(sent=self.forms,
                                                                    vocab=vocab_word_emb,
                                                                    unk=UNK)])

    def set_mark_ids(self):
        mark_ids = [[0 for _ in xrange(self.n_words)] for _ in xrange(self.n_prds)]
        for i, prd_index in enumerate(self.prd_indices):
            mark_ids[i][prd_index] = 1
        self.mark_ids = array(mark_ids)

    def set_label_ids(self, vocab_label):
        label_ids = []
        assert len(self.prd_indices) == len(self.prd_props)
        for prd_index, props in zip(self.prd_indices, self.prd_props):
            y = convert_str_to_id(sent=props,
                                  vocab=vocab_label,
                                  unk=UNDER_BAR)
            label_ids.append(y)
        self.label_ids = array(label_ids)

    def set_label_id_matrix(self, vocab_label):
        label_ids = []
        for index in xrange(self.n_words):
            if index in self.prd_indices:
                prd_i = self.prd_indices.index(index)
                y = convert_str_to_id(sent=self.prd_props[prd_i],
                                      vocab=vocab_label,
                                      unk=UNDER_BAR)
            else:
                y = [0 for _ in xrange(self.n_words)]
            label_ids.append(y)
        self.label_id_matrix = array(label_ids)
