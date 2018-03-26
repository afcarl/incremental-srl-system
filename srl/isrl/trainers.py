from . import write, Trainer


class MulSeqTrainer(Trainer):
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
                                 vocab_lemma=None,
                                 vocab_label=vocab_label)

        if argv.action == 'shift':
            self.model_api.set_train_shift_func()
            self.model_api.set_predict_shift_func()
            self._train_shift_model(train_samples, dev_samples)
        else:
            self.model_api.set_train_label_func()
            self.model_api.set_validate_label_func()
            self._train_label_model(train_samples, dev_samples)

    def _train_shift_model(self, train_samples, dev_samples):
        if dev_samples:
            dev_batches = self.preprocessor.make_batches(dev_samples)
            dev_batch_x, dev_batch_shift = self.preprocessor.separate_x_and_y_in_batch(dev_batches, index=2)
        else:
            dev_batch_x = dev_batch_shift = []

        write('\nTRAIN START')
        f1_history = {}
        best_dev_f = 0.0
        best_epoch = -1

        if self.argv.load_param:
            write('\nEpoch: 0 (Using the Pre-trained Params)')
            if dev_batch_x:
                write('  DEV')
                print '\t',
                dev_batch_shift_pred = self.model_api.predict_shift(dev_batch_x)
                best_dev_f = self.evaluator.f_measure_for_shift(dev_batch_shift, dev_batch_shift_pred)

        for epoch in xrange(self.argv.epoch):
            write('\nEpoch: %d' % (epoch + 1))
            write('  TRAIN')
            print '\t',

            train_batches = self.preprocessor.make_batches(train_samples)
            self.model_api.train_shift_model(train_batches)

            if dev_batch_x:
                write('  DEV')
                print '\t',
                dev_batch_shift_pred = self.model_api.predict_shift(dev_batch_x)
                dev_f = self.evaluator.f_measure_for_shift(dev_batch_shift, dev_batch_shift_pred)

                if best_dev_f < dev_f:
                    best_dev_f = dev_f
                    best_epoch = epoch
                    f1_history[best_epoch + 1] = best_dev_f

                    if self.argv.save:
                        self.model_api.save_shift_model_params()

            self._show_score_history(f1_history)

    def _train_label_model(self, train_samples, dev_samples):
        if dev_samples:
            dev_batches = self.preprocessor.make_batches(dev_samples)
        else:
            dev_batches = []

        write('\nTRAIN START')
        f1_history = {}
        best_dev_f = 0.0
        best_epoch = -1

        if self.argv.load_param:
            write('\nEpoch: 0 (Using the Pre-trained Params)')
            if dev_batches:
                write('  DEV')
                print '\t',
                best_dev_f = self.model_api.predict_label(dev_batches)

        for epoch in xrange(self.argv.epoch):
            write('\nEpoch: %d' % (epoch + 1))
            write('  TRAIN')
            print '\t',

            train_batches = self.preprocessor.make_batches(train_samples)
            self.model_api.train_label_model(train_batches)

            if dev_batches:
                write('  DEV')
                print '\t',
                dev_f = self.model_api.validate_label(dev_batches)

                if best_dev_f < dev_f:
                    best_dev_f = dev_f
                    best_epoch = epoch
                    f1_history[best_epoch + 1] = best_dev_f

                    if self.argv.save:
                        self.model_api.save_label_model_params()

            self._show_score_history(f1_history)
