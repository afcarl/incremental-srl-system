import sys
import time

import numpy as np
import theano
import theano.tensor as T

from ..utils import write
from ..nn import L2Regularizer, get_optimizer
from ..lp import ModelAPI
from models import PIModel


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
        objective = loss + L2Regularizer()(self.argv.lr, self.model.params)
        grads = T.grad(objective, self.model.params)
        self.optimizer = get_optimizer(argv=self.argv)
        updates = self.optimizer(grads=grads, params=self.model.params)

        self.train_func = theano.function(
            inputs=self.model.inputs + [prd_true],
            outputs=[objective,
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

    def predict_online(self, sample):
        return self.pred_func(*sample)
