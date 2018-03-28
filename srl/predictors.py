from utils import write, Vocab, CoNLL09Loader, EmbeddingLoader, CoNLL09Saver


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
        raise NotImplementedError


class LPPredictor(Predictor):
    def run(self):
        argv = self.argv

        ################
        # Load dataset #
        ################
        write('\n\tLoading Dataset...')
        corpus = self.loader.load(path=argv.test_data,
                                  data_size=argv.data_size)
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


class PIPredictor(Predictor):
    def run(self):
        argv = self.argv

        ################
        # Load dataset #
        ################
        write('\n\tLoading Dataset...')
        corpus = self.loader.load(path=argv.test_data,
                                  data_size=argv.data_size)
        sents = self.preprocessor.make_sents(corpus=corpus)

        #################
        # Load init emb #
        #################
        if argv.init_emb:
            write('\n\tLoading Embeddings...')
            word_list_emb, init_emb = self.emb_loader.load(argv.init_emb)
            write('\n\tEmb Vocab Size: %d' % len(word_list_emb))
        else:
            word_list_emb = init_emb = None

        ##############
        # Make words #
        ##############
        if argv.load_word:
            word_key_value = self.loader.load_key_value_format(argv.load_word)
            vocab_word_corpus = self.make_vocab_from_ids(word_key_value)
        else:
            vocab_word_corpus = None

        if word_list_emb:
            vocab_word_emb = self.make_vocab_word(word_list=word_list_emb)
        else:
            vocab_word_emb = None

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
                                 vocab_word_emb=vocab_word_emb)
        self.model_api.load_params()
        self.model_api.set_pred_func()

        ###########
        # Testing #
        ###########
        labels_pred = self.model_api.predict(batches)
