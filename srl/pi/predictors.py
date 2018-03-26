from ..lp import Predictor
from ..utils import write


class PIPredictor(Predictor):
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

