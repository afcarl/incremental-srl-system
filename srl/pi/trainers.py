from ..lp import Trainer
from ..utils import write


class PITrainer(Trainer):
    def run(self):
        argv = self.argv

        train_corpus, dev_corpus = self._load_corpus(argv=argv)
        train_corpus, dev_corpus = self._make_sents(train_corpus=train_corpus,
                                                    dev_corpus=dev_corpus)

        vocab_word_emb, init_emb = self._get_init_emb()
        vocab_word_corpus = self._get_vocab_corpus(train_corpus)

        train_corpus, dev_corpus = self._set_sent_params(train_corpus=train_corpus,
                                                         dev_corpus=dev_corpus,
                                                         vocab_word_corpus=vocab_word_corpus,
                                                         vocab_word_emb=vocab_word_emb)
        train_samples, dev_samples = self._get_samples(train_corpus, dev_corpus)
        self._count_batches(train_samples, dev_samples)

        self.model_api.set_model(init_emb=init_emb,
                                 vocab_word_corpus=vocab_word_corpus,
                                 vocab_word_emb=vocab_word_emb,
                                 vocab_label=None)
        if self.argv.load_param:
            self.model_api.load_params()
        self.model_api.set_train_func()
        self.model_api.set_pred_func()

        self._train(train_samples=train_samples,
                    dev_samples=dev_samples)

    def _train(self, train_samples, dev_samples, vocab_label_dev=None):
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
                best_dev_f = self.evaluator.f_measure_for_pi(y_true=dev_batch_y,
                                                             y_pred=dev_batch_y_pred)

        for epoch in xrange(self.argv.epoch):
            write('\nEpoch: %d' % (epoch + 1))
            write('  TRAIN')

            if self.argv.halve_lr:
                if epoch > 19 and (epoch % 20) == 0:
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
                dev_f = self.evaluator.f_measure_for_pi(y_true=dev_batch_y,
                                                        y_pred=dev_batch_y_pred)

                if best_dev_f < dev_f:
                    best_dev_f = dev_f
                    best_epoch = epoch
                    f1_history[best_epoch + 1] = best_dev_f

                    if self.argv.save:
                        self.model_api.save_params()

            self._show_score_history(f1_history)
