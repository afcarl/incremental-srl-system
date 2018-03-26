from . import write, Evaluator, CoNLL09Loader, EmbeddingLoader, Saver


class Trainer(object):
    def __init__(self, argv, preprocessor, model_api):
        self.argv = argv
        self.emb_loader = EmbeddingLoader(argv)
        self.evaluator = Evaluator(argv)
        self.loader = CoNLL09Loader(argv)
        self.saver = Saver(argv)
        self.preprocessor = preprocessor(argv)
        self.model_api = model_api(argv)

    @staticmethod
    def _show_score_history(history):
        write('\n\tF1 HISTORY')
        for k, v in sorted(history.items()):
            write('\t- EPOCH-{:d}  \tBEST DEV {:>7.2%}'.format(k, v))
        write('\n')

    def _load_corpus(self, argv):
        write('\n\tLoading Dataset...')
        train_corpus = self.loader.load(path=argv.train_data,
                                        data_size=argv.data_size,
                                        is_test=False)
        if argv.dev_data:
            dev_corpus = self.loader.load(path=argv.dev_data,
                                          data_size=argv.data_size,
                                          is_test=False)
        else:
            dev_corpus = []
        write('\t- Train Sents: %d' % len(train_corpus))
        write('\t- Dev   Sents: %d' % len(dev_corpus))
        return train_corpus, dev_corpus

    def _make_sents(self, train_corpus, dev_corpus):
        train_corpus = self.preprocessor.make_sents(train_corpus)
        if dev_corpus:
            dev_corpus = self.preprocessor.make_sents(dev_corpus)
        return train_corpus, dev_corpus

    def _get_init_emb(self):
        if self.argv.init_emb:
            write('\n\tLoading Initial Embeddings...')
            word_list_emb, init_emb = self.emb_loader.load(self.argv.init_emb)
            vocab_word_emb = self.preprocessor.make_vocab_word(word_list_emb)
            write('\n\t# Embedding Words: %d' % vocab_word_emb.size())
        else:
            vocab_word_emb = init_emb = None
        return vocab_word_emb, init_emb

    def _get_vocab_corpus(self, corpus):
        argv = self.argv

        if argv.unuse_word_corpus:
            return None

        word_list_corpus = self.preprocessor.make_word_list(corpus=corpus,
                                                            cut_word=argv.cut_word)
        vocab_word_corpus = self.preprocessor.make_vocab_word(word_list=word_list_corpus)

        if argv.save:
            if vocab_word_corpus:
                if argv.output_fn:
                    fn = 'word_ids.' + argv.output_fn
                else:
                    fn = 'word_ids.size-%d' % vocab_word_corpus.size()
                values, keys = map(lambda x: x, zip(*enumerate(vocab_word_corpus.i2w)))
                self.saver.save_key_value_format(fn=fn, keys=keys, values=values)

        write('\n\t# Words: %d' % vocab_word_corpus.size())
        return vocab_word_corpus

    def _get_vocab_label(self, train_corpus, dev_corpus):
        argv = self.argv

        vocab_label_train = self.preprocessor.make_vocab_label(corpus=train_corpus,
                                                               vocab_label_tmp=None,
                                                               cut_label=argv.cut_label)
        if dev_corpus:
            vocab_label_dev = self.preprocessor.make_vocab_label(corpus=dev_corpus,
                                                                 vocab_label_tmp=vocab_label_train)
        else:
            vocab_label_dev = None

        if argv.save:
            if argv.output_fn:
                fn_d = 'label_ids.' + argv.output_fn
            else:
                fn_d = 'label_ids.size-%d' % vocab_label_train.size()
            values_d, keys_d = map(lambda x: x, zip(*enumerate(vocab_label_train.i2w)))
            self.saver.save_key_value_format(fn=fn_d, keys=keys_d, values=values_d)

        write('\t# Labels %d' % vocab_label_train.size())
        print "\t%s" % str(vocab_label_train.i2w)
        return vocab_label_train, vocab_label_dev

    def _set_sent_params(self, train_corpus, dev_corpus,
                         vocab_word_corpus, vocab_word_emb,
                         vocab_label=None, vocab_label_dev=None):
        train_corpus = self.preprocessor.set_sent_params(corpus=train_corpus,
                                                         vocab_word_corpus=vocab_word_corpus,
                                                         vocab_word_emb=vocab_word_emb,
                                                         vocab_label=vocab_label)
        if dev_corpus:
            dev_corpus = self.preprocessor.set_sent_params(corpus=dev_corpus,
                                                           vocab_word_corpus=vocab_word_corpus,
                                                           vocab_word_emb=vocab_word_emb,
                                                           vocab_label=vocab_label_dev)
        return train_corpus, dev_corpus

    def _get_samples(self, train_corpus, dev_corpus):
        write('\n\tMaking Samples...')
        train_samples = self.preprocessor.make_samples(corpus=train_corpus)
        if dev_corpus:
            dev_samples = self.preprocessor.make_samples(corpus=dev_corpus)
        else:
            dev_samples = []
        write('\t- Train Samples: %d' % len(train_samples))
        write('\t- Dev   Samples: %d' % len(dev_samples))
        return train_samples, dev_samples

    def _count_batches(self, train_samples, dev_samples):
        write('\n\tMaking Batches...')
        train_batches = self.preprocessor.make_batches(samples=train_samples)
        if dev_samples:
            dev_batches = self.preprocessor.make_batches(samples=dev_samples)
        else:
            dev_batches = []
        write('\t- Train Batches: %d' % len(train_batches))
        write('\t- Dev   Batches: %d' % len(dev_batches))

    def run(self):
        argv = self.argv

        train_corpus, dev_corpus = self._load_corpus(argv=argv)
        train_corpus, dev_corpus = self._make_sents(train_corpus=train_corpus,
                                                    dev_corpus=dev_corpus)

        vocab_word_emb, init_emb = self._get_init_emb()
        vocab_word_corpus = self._get_vocab_corpus(train_corpus)
        vocab_label, vocab_label_dev = self._get_vocab_label(train_corpus, dev_corpus)

        train_corpus, dev_corpus = self._set_sent_params(train_corpus=train_corpus,
                                                         dev_corpus=dev_corpus,
                                                         vocab_word_corpus=vocab_word_corpus,
                                                         vocab_word_emb=vocab_word_emb,
                                                         vocab_label=vocab_label,
                                                         vocab_label_dev=vocab_label_dev)
        train_samples, dev_samples = self._get_samples(train_corpus, dev_corpus)
        self._count_batches(train_samples, dev_samples)

        self.model_api.set_model(init_emb=init_emb,
                                 vocab_word_corpus=vocab_word_corpus,
                                 vocab_word_emb=vocab_word_emb,
                                 vocab_label=vocab_label)
        if self.argv.load_param:
            self.model_api.load_params()
        self.model_api.set_train_func()
        self.model_api.set_pred_func()

        self._train(train_samples=train_samples,
                    dev_samples=dev_samples,
                    vocab_label_dev=vocab_label_dev)

    def _train(self, train_samples, dev_samples, vocab_label_dev):
        if dev_samples:
            dev_batches = self.preprocessor.make_batches(dev_samples)
            dev_batch_x, dev_batch_y = self.preprocessor.separate_x_and_y_in_batch(dev_batches)
        else:
            dev_batch_x = dev_batch_y = []

        write('\nTRAIN START')
        f1_history = {}
        best_dev_f = 0.0
        best_epoch = -1

        if self.argv.load_param:
            write('\nEpoch: 0 (Using the Pre-trained Params)')
            if dev_batch_x:
                write('  DEV')
                print '\t',
                dev_batch_y_pred = self.model_api.predict(dev_batch_x)
                best_dev_f = self.evaluator.f_measure_for_label(y_true=dev_batch_y,
                                                                y_pred=dev_batch_y_pred,
                                                                vocab_label=vocab_label_dev)

        for epoch in xrange(self.argv.epoch):
            write('\nEpoch: %d' % (epoch + 1))
            write('  TRAIN')

            if self.argv.halve_lr:
                if epoch > 49 and (epoch % 25) == 0:
                    lr = self.model_api.optimizer.lr.get_value(borrow=True)
                    self.model_api.optimizer.lr.set_value(lr * 0.5)
                    write('  ### HALVE LEARNING RATE: %f -> %f' % (lr, lr * 0.5))
            print '\t',

            train_batches = self.preprocessor.make_batches(train_samples)
            self.model_api.train(train_batches)

            if dev_batch_x:
                write('  DEV')
                print '\t',
                dev_batch_y_pred = self.model_api.predict(dev_batch_x)
                dev_f = self.evaluator.f_measure_for_label(y_true=dev_batch_y,
                                                           y_pred=dev_batch_y_pred,
                                                           vocab_label=vocab_label_dev)

                if best_dev_f < dev_f:
                    best_dev_f = dev_f
                    best_epoch = epoch
                    f1_history[best_epoch + 1] = best_dev_f

                    if self.argv.save:
                        self.model_api.save_params()

            self._show_score_history(f1_history)
