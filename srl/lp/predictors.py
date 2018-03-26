from . import write, Vocab, CoNLL09Loader, EmbeddingLoader, CoNLL09Saver
from ..pi.model_api import PIModelAPI


class Predictor(object):
    def __init__(self, argv, model_api, preprocessor):
        self.argv = argv
        self.emb_loader = EmbeddingLoader(argv)
        self.loader = CoNLL09Loader(argv)
        self.saver = CoNLL09Saver(argv)
        self.preprocessor = preprocessor(argv)
        self.model_api = model_api(argv)

    def make_vocab_word(self, word_list):
        vocab_word = self.preprocessor.make_vocab_word(word_list)
        write('\n\tVocab Size: %d' % vocab_word.size())
        return vocab_word

    @staticmethod
    def make_vocab_from_ids(key_value_format):
        vocab = Vocab()
        for key, value in key_value_format:
            vocab.add_word(key)
        return vocab

    def run(self):
        argv = self.argv

        ################
        # Load dataset #
        ################
        write('\n\tLoading Dataset...')
        corpus = self.loader.load(path=argv.test_data,
                                  data_size=argv.data_size,
                                  is_test=True)
        sents = self.preprocessor.make_sents(corpus=corpus)

        #################
        # Load init emb #
        #################
        if argv.init_emb:
            write('\n\tLoading Embeddings...')
            word_list_emb, init_emb = self.emb_loader.load(argv.init_emb)
            vocab_word_emb = self.make_vocab_word(word_list_emb)
            write('\n\tEmb Vocab Size: %d' % len(word_list_emb))
        else:
            vocab_word_emb = init_emb = None

        ##############
        # Make words #
        ##############
        if argv.load_word:
            word_key_value = self.loader.load_key_value_format(argv.load_word)
            vocab_word_corpus = self.make_vocab_from_ids(word_key_value)
        else:
            vocab_word_corpus = None

        ###############
        # Make labels #
        ###############
        label_key_value = self.loader.load_key_value_format(argv.load_label)
        vocab_label = self.make_vocab_from_ids(label_key_value)

        ################
        # Make samples #
        ################
        write('\n\tMaking Samples...')
        samples = self.preprocessor.set_sent_params(corpus=sents,
                                                    vocab_word_corpus=vocab_word_corpus,
                                                    vocab_word_emb=vocab_word_emb,
                                                    vocab_label=None)
        batches = self.preprocessor.make_test_batches(corpus=samples)
        write('\t- Samples: %d' % len(batches))

        #############
        # Model API #
        #############
        self.model_api.set_model(init_emb=init_emb,
                                 vocab_word_corpus=vocab_word_corpus,
                                 vocab_word_emb=vocab_word_emb,
                                 vocab_label=vocab_label)
        self.model_api.load_params(argv.load_param)
        self.model_api.set_pred_func()

        ###########
        # Testing #
        ###########
        labels_pred = self.model_api.predict(batches)
        self.saver.save_predicted_prop(corpus=corpus,
                                       results=labels_pred,
                                       vocab_label=vocab_label)

    def run_online(self):
        argv = self.argv

        #################
        # Load init emb #
        #################
        if argv.init_emb:
            write('\n\tLoading Embeddings...')
            word_list_emb, init_emb = self.emb_loader.load(argv.init_emb)
            vocab_word_emb = self.make_vocab_word(word_list=word_list_emb)
            write('\n\tEmb Vocab Size: %d' % len(word_list_emb))
        else:
            vocab_word_emb = init_emb = None

        ##############
        # Make words #
        ##############
        if argv.load_word:
            word_key_value = self.loader.load_key_value_format(argv.load_word)
            vocab_word_corpus = self.make_vocab_from_ids(word_key_value)
        else:
            vocab_word_corpus = None

        if argv.load_pi_word:
            word_key_value = self.loader.load_key_value_format(argv.load_word)
            vocab_word_corpus_pi = self.make_vocab_from_ids(word_key_value)
        else:
            vocab_word_corpus_pi = None

        ###############
        # Make labels #
        ###############
        label_key_value = self.loader.load_key_value_format(argv.load_label)
        vocab_label = self.make_vocab_from_ids(label_key_value)

        #############
        # Model API #
        #############
        pi_model_api = PIModelAPI(argv)
        pi_model_api.set_model(init_emb=init_emb,
                               vocab_word_corpus=vocab_word_corpus_pi,
                               vocab_word_emb=vocab_word_emb,
                               vocab_label=None)
        pi_model_api.load_params(argv.load_pi_param)
        pi_model_api.set_pred_func()

        self.model_api.set_model(init_emb=init_emb,
                                 vocab_word_corpus=vocab_word_corpus,
                                 vocab_word_emb=vocab_word_emb,
                                 vocab_label=vocab_label)
        self.model_api.load_params(argv.load_param)
        self.model_api.set_pred_func()

        self._predict(pi_model_api, vocab_word_corpus, vocab_word_emb, vocab_label)

    def _predict(self, pi_model_api, vocab_word_corpus, vocab_word_emb, vocab_label):
        while True:
            print '\nInput a tokenized sentence.'
            input_sent = raw_input('>>>  ')
            input_sent = input_sent.split()

            sample = self.preprocessor.make_sample_from_input_sent(input_sent, vocab_word_corpus, vocab_word_emb)
            mark_pred = pi_model_api.predict_online(sample)
            sample = self.preprocessor.add_mark_to_sample(sample, mark_pred)
            labels_pred = self.model_api.predict_online(sample)

            self._show_result(input_sent, vocab_label, sample[-1], labels_pred)

    @staticmethod
    def _show_result(input_sent, vocab_label, mark_pred, labels_pred):
        for marks_i, labels_i in zip(mark_pred, labels_pred):
            prd_index = list(marks_i).index(1)
            prd = input_sent[prd_index]
            labels = [vocab_label.get_word(label_id) for label_id in labels_i]
            labels[prd_index] = 'PRD'
            text = map(lambda (x, y): '%s/%s' % (x, y), zip(input_sent, labels))
            print 'PRD:%s  %s' % (prd, ' '.join(text))
