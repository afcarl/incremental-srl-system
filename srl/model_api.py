import sys
import time
import math
import cPickle
import gzip

import numpy as np
import theano
import theano.tensor as T

from utils import write
from nn import categorical_accuracy, get_optimizer, L2Regularizer
from models import LPModel, PIModel


class ModelAPI(object):
    def __init__(self, argv):
        self.argv = argv

        self.model = None
        self.train_func = None
        self.pred_func = None

        self.vocabs = []
        self.vocab_word_corpus = None
        self.vocab_word_emb = None
        self.vocab_label = None

        self.input_dim = None
        self.hidden_dim = None
        self.output_dim = None

        self.optimizer = None

    def set_model(self):
        raise NotImplementedError

    def _show_model_config(self):
        model = self.model
        write('\nModel configuration')
        write('\t- Vocab Size: {}'.format(
            self.vocab_word_corpus.size() if self.vocab_word_corpus else self.vocab_word_emb.size()))
        write('\t- Input  Dim: {}'.format(self.input_dim))
        write('\t- Hidden Dim: {}'.format(self.hidden_dim))
        write('\t- Output Dim: {}'.format(self.output_dim))
        write('\t- Parameters: {}\n'.format(sum(len(x.get_value(borrow=True).ravel()) for x in model.params)))
        write('\tMODEL: {}'.format(" -> ".join([l.name for l in self.model.layers])))

    @staticmethod
    def save_data(data, fn):
        with gzip.open(fn + '.pkl.gz', 'wb') as gf:
            cPickle.dump(data, gf, cPickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_data(fn):
        with gzip.open(fn, 'rb') as gf:
            return cPickle.load(gf)

    def save_params(self):
        argv = self.argv
        if argv.output_fn:
            fn = 'param/param.' + argv.output_fn
        else:
            fn = 'param/param.' + argv.task

        params = [p.get_value(borrow=True) for p in self.model.params]
        self.save_data(data=params, fn=fn)

    def load_params(self, path):
        params = self.load_data(path)
        assert len(self.model.params) == len(params)
        for p1, p2 in zip(self.model.params, params):
            p1.set_value(p2)


class LPModelAPI(ModelAPI):
    def set_model(self, **kwargs):
        argv = self.argv

        self.vocab_word_corpus = kwargs['vocab_word_corpus']
        self.vocab_word_emb = kwargs['vocab_word_emb']
        self.vocab_label = kwargs['vocab_label']
        self.vocabs = [self.vocab_word_corpus, self.vocab_word_emb, self.vocab_label]

        self.input_dim = kwargs['init_emb'].shape[1] if kwargs['init_emb'] is not None else argv.emb_dim
        self.hidden_dim = argv.hidden_dim
        self.output_dim = kwargs['vocab_label'].size()

        inputs = self._set_inputs()
        self.model = LPModel()
        self.model.compile(
            inputs=inputs,
            vocab_word_corpus_size=self.vocab_word_corpus.size() if self.vocab_word_corpus else 0,
            vocab_word_emb_size=self.vocab_word_emb.size() if self.vocab_word_emb else 0,
            input_dim=[self.input_dim, 5],
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            n_layers=argv.n_layers,
            rnn_unit=argv.rnn_unit,
            batch_size=argv.batch_size,
            init_emb=kwargs['init_emb'],
            init_emb_fix=argv.init_emb_fix,
            drop_rate=argv.drop_rate
        )
        self._show_model_config()

    def _set_inputs(self):
        x = []
        if self.vocab_word_corpus:
            x.append(T.imatrix('x_word_corpus'))
        if self.vocab_word_emb:
            x.append(T.imatrix('x_word_emb'))
        x.append(T.imatrix('x_mark'))
        assert len(x) > 1
        return x

    def set_train_func(self):
        write('\nBuilding an lp train func...')
        y_label = T.imatrix('y')

        label_proba = self.model.calc_label_proba(self.model.inputs)
        label_pred = self.model.argmax_label_proba(label_proba)
        true_label_path_score = self.model.calc_label_path_score(label_proba, y_label)

        cost = - T.mean(true_label_path_score) + L2Regularizer()(alpha=self.argv.reg, params=self.model.params)
        grads = T.grad(cost=cost, wrt=self.model.params)
        self.optimizer = get_optimizer(argv=self.argv)
        updates = self.optimizer(grads=grads, params=self.model.params)

        self.train_func = theano.function(
            inputs=self.model.inputs + [y_label],
            outputs=[cost,
                     categorical_accuracy(y_true=y_label, y_pred=label_pred),
                     label_pred.flatten(),
                     y_label.flatten()
                     ],
            updates=updates,
            mode='FAST_RUN'
        )

    def set_pred_func(self):
        write('\nBuilding an lp predict func...')
        label_proba = self.model.calc_label_proba(self.model.inputs)
        y_pred = self.model.argmax_label_proba(label_proba)

        self.pred_func = theano.function(
            inputs=self.model.inputs,
            outputs=y_pred,
            mode='FAST_RUN'
        )

    def train(self, batches):
        batch_indices = range(len(batches))
        np.random.shuffle(batch_indices)

        start = time.time()
        n_samples = 0.
        loss_total = 0.
        crr_total = 0.

        p_total = 0.
        r_total = 0.
        correct = 0.

        self.model.feat_layer.is_train.set_value(1)
        for index, b_index in enumerate(batch_indices):
            if (index + 1) % 100 == 0:
                print '%d' % (index + 1),
                sys.stdout.flush()

            inputs = batches[b_index]
            loss, crr, y_pred, y_true = self.train_func(*inputs)

            loss_total += loss
            crr_total += crr
            n_samples += len(inputs[0]) * len(inputs[0][0])

            p_total += sum([1 for y_i in y_pred if y_i > 0])
            r_total += sum([1 for y_i in y_true if y_i > 0])
            correct += sum([1 for y_true_i, y_pred_i in zip(y_true, y_pred) if y_true_i == y_pred_i > 0])

            if math.isnan(loss):
                write('\n\nNAN: Index: %d\n' % (index + 1))
                exit()

        avg_loss = loss_total / float(len(batches))
        acc = crr_total / n_samples
        p = correct / p_total if p_total > 0 else 0.
        r = correct / r_total if r_total > 0 else 0.
        f = (2 * p * r) / (p + r) if p + r > 0 else 0.

        write('\n\tTime: %f seconds' % (time.time() - start))
        write('\tAverage Negative Log Likelihood: %f(%f/%d)' % (avg_loss, loss_total, len(batches)))
        write('\tAccuracy: %f (%d/%d)' % (acc, crr_total, n_samples))
        write('\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})\n'.format(
            f, p, int(correct), int(p_total), r, int(correct), int(r_total)))

    def predict(self, batches):
        """
        :param batches: 1D: n_batches, 2D: n_words; elem=(x_w, x_m)
        :return: y: 1D: n_batches, 2D: batch_size; elem=(y_pred(1D:n_words), y_proba(float))
        """
        start = time.time()
        y = []

        self.model.feat_layer.is_train.set_value(0)
        for index, inputs in enumerate(batches):
            if (index + 1) % 1000 == 0:
                print '%d' % (index + 1),
                sys.stdout.flush()

            if len(inputs) == 0:
                y_pred = []
            else:
                y_pred = self.pred_func(*inputs)
            y.append(y_pred)

        write('\n\tTime: %f seconds' % (time.time() - start))
        return y


class PIModelAPI(ModelAPI):
    def set_model(self, **kwargs):
        write('\nCompiling a computational graph...')
        argv = self.argv

        self.vocab_word_corpus = kwargs['vocab_word_corpus']
        self.vocab_word_emb = kwargs['vocab_word_emb']

        self.input_dim = kwargs['init_emb'].shape[1] if kwargs['init_emb'] is not None else argv.emb_dim
        self.hidden_dim = argv.hidden_dim

        inputs = self._set_inputs()
        self.model = PIModel()
        self.model.compile(inputs=inputs,
                           vocab_word_corpus_size=self.vocab_word_corpus.size() if self.vocab_word_corpus else 0,
                           vocab_word_emb_size=self.vocab_word_emb.size() if self.vocab_word_emb else 0,
                           input_dim=self.input_dim,
                           hidden_dim=self.hidden_dim,
                           n_layers=argv.n_layers,
                           rnn_unit=argv.rnn_unit,
                           batch_size=argv.batch_size,
                           init_emb=kwargs['init_emb'],
                           init_emb_fix=argv.init_emb_fix,
                           drop_rate=argv.drop_rate)
        self._show_model_config()

    def _set_inputs(self):
        x = []
        if self.vocab_word_corpus:
            x.append(T.imatrix('x_word_corpus'))
        if self.vocab_word_emb:
            x.append(T.imatrix('x_word_emb'))
        assert len(x) > 0
        return x

    def set_train_func(self):
        write('\nBuilding a pi train func...')

        prd_true = T.imatrix('prd_true')
        prd_proba = self.model.calc_proba(self.model.inputs)
        prd_pred = self.model.binary_prediction(prd_proba)

        crr, true_total, pred_total = self.model.calc_correct_predictions(y_true=prd_true,
                                                                          y_pred=prd_pred)
        loss = self.model.get_binary_loss(prd_true, prd_proba)

        cost = loss + L2Regularizer()(alpha=self.argv.lr, params=self.model.params)
        grads = T.grad(cost=cost, wrt=self.model.params)
        self.optimizer = get_optimizer(argv=self.argv)
        updates = self.optimizer(grads=grads, params=self.model.params)

        self.train_func = theano.function(
            inputs=self.model.inputs + [prd_true],
            outputs=[cost,
                     crr,
                     true_total,
                     pred_total
                     ],
            updates=updates,
            mode='FAST_RUN',
        )

    def set_pred_func(self):
        write('\nBuilding a pi predict func...')

        prd_proba = self.model.calc_proba(self.model.inputs)
        prd_pred = self.model.binary_prediction(prd_proba)

        self.pred_func = theano.function(
            inputs=self.model.inputs,
            outputs=prd_pred,
            mode='FAST_RUN',
        )

    def train(self, batches):
        loss_total = 0.
        crr = 0.
        true_total = 0.
        pred_total = 0.

        start = time.time()
        batch_indices = range(len(batches))
        np.random.shuffle(batch_indices)

        self.model.feat_layer.is_train.set_value(1)
        for index, b_index in enumerate(batch_indices):
            if (index + 1) % 100 == 0:
                print '%d' % (index + 1),
                sys.stdout.flush()

            batch = batches[b_index]
            loss_i, crr_i, true_i, pred_i = self.train_func(*batch)

            loss_total += loss_i
            crr += crr_i
            true_total += true_i
            pred_total += pred_i

        avg_loss = loss_total / float(len(batches))
        precision = crr / pred_total if pred_total > 0 else 0.
        recall = crr / true_total if true_total > 0 else 0.
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.

        write('\n\tTime: %f seconds' % (time.time() - start))
        write('\tAverage Negative Log Likelihood: %f' % avg_loss)
        write('\tLabel: F1:%f\tP:%f(%d/%d)\tR:%f(%d/%d)' % (f1, precision, crr, pred_total, recall, crr, true_total))

    def predict(self, batches):
        labels = []
        start = time.time()

        self.model.feat_layer.is_train.set_value(0)
        for index, batch in enumerate(batches):
            if (index + 1) % 100 == 0:
                print '%d' % (index + 1),
                sys.stdout.flush()

            if len(batch) == 0:
                label_pred = []
            else:
                label_pred = self.pred_func(*batch)
            labels.append(label_pred)
        write('\n\tTime: %f seconds' % (time.time() - start))
        return labels
