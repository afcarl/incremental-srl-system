from ..utils import write
from ..predictors import Predictor


class ISRLPredictor(Predictor):
    def run(self):
        argv = self.argv
        pi_args = self.loader.load_data(argv.load_pi_args)
        lp_args = self.loader.load_data(argv.load_lp_args)

        ################
        # Load dataset #
        ################
        write('\n\tLoading Dataset...')
        corpus = self.loader.load(path=argv.test_data,
                                  data_size=argv.data_size)
        corpus = self.preprocessor.make_sents(corpus=corpus,
                                              marked_prd=True if argv.action == 'label' else False)

        #################
        # Load init emb #
        #################
        if argv.init_emb:
            write('\n\tLoading Embeddings...')
            word_list_emb, init_emb = self.emb_loader.load(argv.init_emb)
            vocab_word_emb = self.make_vocab_word(word_list=word_list_emb)
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
        samples = self.preprocessor.set_sent_params(corpus=corpus,
                                                    vocab_word_corpus=vocab_word_corpus,
                                                    vocab_word_emb=vocab_word_emb,
                                                    vocab_label=None)
        write('\t- Samples: %d' % len(samples))

        #############
        # Model API #
        #############
        self.model_api.set_model(pi_args=pi_args,
                                 lp_args=lp_args,
                                 init_emb=init_emb,
                                 vocab_word_corpus=vocab_word_corpus,
                                 vocab_word_emb=vocab_word_emb,
                                 vocab_label=vocab_label)

        if argv.action == 'pi_lp':
            self.predict(samples)
        else:
            self.predict_given_gold_prds(samples)

    def run_cmd_mode(self):
        argv = self.argv
        pi_args = self.loader.load_data(argv.load_pi_args)
        lp_args = self.loader.load_data(argv.load_lp_args)

        #################
        # Load init emb #
        #################
        if argv.init_emb:
            write('\n\tLoading Embeddings...')
            word_list_emb, init_emb = self.emb_loader.load(argv.init_emb)
            vocab_word_emb = self.make_vocab_word(word_list=word_list_emb)
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

        #############
        # Model API #
        #############
        self.model_api.set_model(pi_args=pi_args,
                                 lp_args=lp_args,
                                 init_emb=init_emb,
                                 vocab_word_corpus=vocab_word_corpus,
                                 vocab_word_emb=vocab_word_emb,
                                 vocab_label=vocab_label)

        self.model_api.load_shift_model_params()
        self.model_api.load_label_model_params()
        self.model_api.set_predict_online_func()

        self.predict_cmd_mode(vocab_word_corpus, vocab_word_emb, vocab_label)

    def run_server_mode(self):
        argv = self.argv
        pi_args = self.loader.load_data(argv.load_pi_args)
        lp_args = self.loader.load_data(argv.load_lp_args)

        #################
        # Load init emb #
        #################
        if argv.init_emb:
            write('\n\tLoading Embeddings...')
            word_list_emb, init_emb = self.emb_loader.load(argv.init_emb)
            vocab_word_emb = self.make_vocab_word(word_list=word_list_emb)
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

        #############
        # Model API #
        #############
        self.model_api.set_model(pi_args=pi_args,
                                 lp_args=lp_args,
                                 init_emb=init_emb,
                                 vocab_word_corpus=vocab_word_corpus,
                                 vocab_word_emb=vocab_word_emb,
                                 vocab_label=vocab_label)

        self.model_api.load_shift_model_params()
        self.model_api.load_label_model_params()
        self.model_api.set_predict_online_func()

    def predict_given_gold_prds(self, samples):
        self.model_api.load_label_model_params()
        self.model_api.set_predict_given_gold_prds_func()
        batches = self.preprocessor.make_test_batches(samples)

        write('\nPREDICTION START')
        print '\t',
        samples = self.model_api.predict_given_gold_prds(batches)
        self.saver.save_isrl_props(samples, self.model_api.vocab_label)

    def predict(self, samples):
        self.model_api.load_shift_model_params()
        self.model_api.load_label_model_params()
        self.model_api.set_predict_func()

        write('\nPREDICTION START')
        print '\t',
        samples = self.model_api.predict(samples)
        self.saver.save_isrl_props(samples, self.model_api.vocab_label)

    def predict_cmd_mode(self, vocab_word_corpus, vocab_word_emb, vocab_label):
        print '\nInput a tokenized sentence.'
        time_step = 0

        sent = []
        while True:
            word = raw_input('>>>  ')
            sample = self.preprocessor.make_sample_from_input_word(word.lower(),
                                                                   vocab_word_corpus,
                                                                   vocab_word_emb,
                                                                   time_step)
            outputs = self.model_api.predict_online(sample)
            stack_a, stack_p, shift_proba, label_proba, label_pred = outputs

            sent.append(word)
            print " ".join(sent)
            """
            print 'SHIFT PROBA: %f' % shift_proba
            print 'STACK A:'
            print stack_a
            print 'STACK P:'
            print stack_p
            """
            for i, (p, labels) in enumerate(zip(stack_p, label_pred)):
                if p == 0:
                    continue
                print 'PRD:%s ' % sent[i],
                for w_index in xrange(len(sent)):
                    form = sent[w_index]
                    label = vocab_label.get_word(labels[w_index])
                    print '%s/%s' % (form, label),
                print

            time_step += 1

    def predict_server_mode(self, word, time_step):
        sample = self.preprocessor.make_sample_from_input_word(word.lower(),
                                                               self.model_api.vocab_word_corpus,
                                                               self.model_api.vocab_word_emb,
                                                               time_step)
        return self.model_api.predict_online(sample)
