import sys
import time

import numpy as np
import theano
import theano.tensor as T

from . import write, L2Regularizer, get_optimizer, ModelAPI
from models import ISRLSystem, ShiftModel, LabelModel


class ISRLSystemAPI(ModelAPI):
    def __init__(self, argv):
        super(ISRLSystemAPI, self).__init__(argv)
        self.train_shift_func = None
        self.train_label_func = None
        self.predict_shift_func = None
        self.predict_label_func = None
        self.predict_shift_and_label_func = None

    def save_shift_model_params(self):
        argv = self.argv
        if argv.output_fn:
            fn = 'param_shift.' + argv.output_fn
        else:
            fn = 'param_shift.layers-%s.e-%d.h-%d' % (argv.n_layers, argv.emb_dim, argv.hidden_dim)
        params = [p.get_value(borrow=True) for p in self.model.shift_model.params]
        self.save_data(data=params, fn=fn)

    def save_label_model_params(self):
        argv = self.argv
        if argv.output_fn:
            fn = 'param_label.' + argv.output_fn
        else:
            fn = 'param_label.layers-%s.e-%d.h-%d' % (argv.n_layers, argv.emb_dim, argv.hidden_dim)
        params = [p.get_value(borrow=True) for p in self.model.label_model.params]
        self.save_data(data=params, fn=fn)

    def load_shift_model_params(self):
        params = self.load_data(self.argv.load_shift_model_param)
        assert len(self.model.shift_model.params) == len(params)
        for p1, p2 in zip(self.model.shift_model.params, params):
            p1.set_value(p2)

    def load_label_model_params(self):
        params = self.load_data(self.argv.load_label_model_param)
        assert len(self.model.label_model.params) == len(params)
        for p1, p2 in zip(self.model.label_model.params, params):
            p1.set_value(p2)

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

        shift_model = ShiftModel()
        shift_model.compile(
            inputs=inputs,
            vocab_word_corpus_size=self.vocab_word_corpus.size() if self.vocab_word_corpus else 0,
            vocab_word_emb_size=self.vocab_word_emb.size() if self.vocab_word_emb else 0,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_layers=argv.n_layers,
            rnn_unit=argv.rnn_unit,
            batch_size=argv.batch_size,
            init_emb=kwargs['init_emb'],
            init_emb_fix=argv.init_emb_fix,
            drop_rate=argv.drop_rate
        )

        label_model = LabelModel()
        label_model.compile(
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

        self.model = ISRLSystem(inputs, shift_model, label_model)
        self._show_model_config()

    def _show_model_config(self):
        model = self.model
        write('\nModel configuration')
        write('\t- Vocab Size: {}'.format(
            self.vocab_word_corpus.size() if self.vocab_word_corpus else self.vocab_word_emb.size())
        )
        write('\t- Input  Dim: {}'.format(self.input_dim))
        write('\t- Hidden Dim: {}'.format(self.hidden_dim))
        write('\t- Output Dim: {}\n'.format(self.output_dim))

        if self.model.shift_model is not None:
            l_names = [l.name for l in self.model.shift_model.layers]
            n_params = sum(len(x.get_value(borrow=True).ravel()) for x in model.shift_model.params)
            write('\tSHIFT: {}'.format(" -> ".join(l_names)))
            write('\t  - Params: {}'.format(n_params))
        if self.model.label_model is not None:
            l_names = [l.name for l in self.model.label_model.layers]
            n_params = sum(len(x.get_value(borrow=True).ravel()) for x in model.label_model.params)
            write('\tLABEL: {}'.format(" -> ".join(l_names)))
            write('\t  - Params: {}'.format(n_params))

    def _set_inputs(self):
        x = []
        if self.vocab_word_corpus:
            x.append(T.imatrix('x_word_corpus'))
        if self.vocab_word_emb:
            x.append(T.imatrix('x_word_emb'))
        assert len(x) > 0
        return x

    def set_train_shift_func(self):
        write('\nBuilding a train shift func...')

        y_shift = T.imatrix('y_shift')
        shift_proba = self.model.get_shift_proba()
        shift_proba = shift_proba.dimshuffle(1, 0)
        shift_pred = T.gt(shift_proba, 0.5)
        crr, true_total, pred_total = self.model.calc_correct_shifts(shift_true=y_shift,
                                                                     shift_pred=shift_pred)

        loss = self.model.get_shift_loss(y_shift, shift_proba)
        objective = loss
        grads = T.grad(objective, self.model.shift_model.params)
        optimizer = get_optimizer(argv=self.argv)
        updates = optimizer(grads=grads, params=self.model.shift_model.params)

        self.train_shift_func = theano.function(
            inputs=self.model.inputs + [y_shift],
            outputs=[objective,
                     crr,
                     true_total,
                     pred_total
                     ],
            updates=updates,
            mode='FAST_RUN',
        )

    def set_train_label_func(self):
        write('\nBuilding a train label func...')

        y_shift = T.imatrix('y_shift')
        y_label = T.itensor3('y_label')
        label_proba, stack_a, stack_p = self.model.get_label_proba_with_oracle_shift(y_shift)
        y_label_pred = self.model.argmax_label_proba(label_proba)
        crr, true_total, pred_total = self.model.calc_correct_labels(label_true=y_label,
                                                                     label_pred=y_label_pred,
                                                                     prd_marks=y_shift)

        loss = self.model.get_label_loss(y_label, label_proba)
        reg = L2Regularizer()(alpha=self.argv.reg, params=self.model.label_model.params)
        objective = loss + reg
        grads = T.grad(objective, self.model.label_model.params)
        optimizer = get_optimizer(argv=self.argv)
        updates = optimizer(grads=grads, params=self.model.label_model.params)

        self.train_label_func = theano.function(
            inputs=self.model.inputs + [y_shift, y_label],
            outputs=[objective,
                     crr,
                     true_total,
                     pred_total
                     ],
            updates=updates,
            mode='FAST_RUN',
        )

    def set_validate_label_func(self):
        write('\nBuilding a validate label func...')

        y_shift = T.imatrix('y_shift')
        y_label = T.itensor3('y_label')

        label_proba, stack_a, stack_p = self.model.get_label_proba_with_oracle_shift(y_shift)

        # 1D: time_steps, 2D: batch_size * n_words(prd) * n_words(arg); elem=label id
        y_label_pred = self.model.argmax_label_proba(label_proba)

        crr, true_total, pred_total = self.model.calc_correct_labels(label_true=y_label,
                                                                     label_pred=y_label_pred,
                                                                     prd_marks=y_shift)

        self.predict_label_func = theano.function(
            inputs=self.model.inputs + [y_shift, y_label],
            outputs=[crr,
                     true_total,
                     pred_total
                     ],
            mode='FAST_RUN',
        )

    def set_predict_shift_func(self):
        write('\nBuilding a predict shift func...')

        shift_proba = self.model.get_shift_proba()
        shift_proba = shift_proba.dimshuffle(1, 0)
        shift_pred = T.gt(shift_proba, 0.5)

        self.predict_shift_func = theano.function(
            inputs=self.model.inputs,
            outputs=shift_pred,
            mode='FAST_RUN',
        )

    def set_predict_online_shift_func(self):
        write('\nBuilding a predict online shift func...')

        x = []
        if self.vocab_word_corpus:
            x.append(T.ivector('x_word_corpus'))
        if self.vocab_word_emb:
            x.append(T.ivector('x_word_emb'))
        time_step = T.iscalar('time_step')

        stack_a = self.model.update_stack_a(x, time_step)
        shift_proba = self.model.get_shift_proba_online(stack_a, time_step)
        shift_id = self.model.get_shift_id(shift_proba)
        stack_p = self.model.update_stack_p(shift_id, time_step)

        updates = [(self.model.stack_a, stack_a),
                   (self.model.stack_p, stack_p)]

        self.predict_shift_func = theano.function(
            inputs=x + [time_step],
            outputs=[stack_a[:, 0].flatten(),
                     T.sum(stack_p[:, 0], axis=0),
                     shift_proba],
            updates=updates,
            mode='FAST_RUN',
        )

    def set_predict_online_shift_and_label_func(self):
        write('\nBuilding a predict online shift and label func...')

        x = []
        if self.vocab_word_corpus:
            x.append(T.ivector('x_word_corpus'))
        if self.vocab_word_emb:
            x.append(T.ivector('x_word_emb'))

        time_step = T.iscalar('time_step')

        stack_a = self.model.update_stack_a(x, time_step)
        shift_proba = self.model.get_shift_proba_online(stack_a, time_step)
        shift_id = self.model.get_shift_id(shift_proba)
        stack_p = self.model.update_stack_p(shift_id, time_step)

        label_proba = self.model.label_model.get_label_proba(stack_a,
                                                             stack_p,
                                                             time_step)
        label_pred = self.model.argmax_label_proba(label_proba)

        updates = [(self.model.stack_a, stack_a),
                   (self.model.stack_p, stack_p)]

        self.predict_shift_and_label_func = theano.function(
            inputs=x + [time_step],
            outputs=[stack_a[:, 0].flatten(),
                     T.sum(stack_p[:, 0], axis=0),
                     shift_proba,
                     label_proba,
                     label_pred],
            updates=updates,
            mode='FAST_RUN',
            on_unused_input='warn'
        )

    def set_predict_label_func(self):
        write('\nBuilding a predict label func...')
        print self.model.inputs

        y_shift = T.imatrix('y_shift')
        label_proba, stack_a, stack_p = self.model.get_label_proba_with_oracle_shift(y_shift)

        # 1D: time_steps, 2D: batch_size * n_words(prd) * n_words(arg); elem=label id
        y_label_pred = self.model.argmax_label_proba(label_proba)

        self.predict_label_func = theano.function(
            inputs=self.model.inputs + [y_shift],
            outputs=y_label_pred,
            mode='FAST_RUN',
        )

    def set_predict_shift_and_label_func(self):
        write('\nBuilding a predict shift and label func...')

        shift_proba, label_proba, stack_a, stack_p = self.model.get_shift_and_label_proba()

        # 1D: time_steps, 2D: batch_size; elem=shift id
        shift_pred = T.gt(shift_proba, 0.5)
        # 1D: time_steps, 2D: batch_size * n_words(prd) * n_words(arg); elem=label id
        label_pred = self.model.argmax_label_proba(label_proba)

        self.predict_shift_and_label_func = theano.function(
            inputs=self.model.inputs,
            outputs=[shift_pred,
                     label_pred,
                     ],
            mode='FAST_RUN',
        )

    def train_shift_model(self, batches):
        loss_total = 0.
        crr = 0.
        true_total = 0.
        pred_total = 0.

        start = time.time()
        batch_indices = range(len(batches))
        np.random.shuffle(batch_indices)
        for index, b_index in enumerate(batch_indices):
            if (index + 1) % 100 == 0:
                print '%d' % (index + 1),
                sys.stdout.flush()

            batch = batches[b_index]
            loss_i, crr_i, true_i, pred_i = self.train_shift_func(*batch[:-1])

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

    def train_label_model(self, batches):
        loss_total = 0.
        crr = 0.
        true_total = 0.
        pred_total = 0.

        start = time.time()
        batch_indices = range(len(batches))
        np.random.shuffle(batch_indices)
        for index, b_index in enumerate(batch_indices):
            if (index + 1) % 100 == 0:
                print '%d' % (index + 1),
                sys.stdout.flush()

            loss_i, crr_i, true_i, pred_i = self.train_label_func(*batches[b_index])

            loss_total += loss_i
            crr += crr_i[-1]
            true_total += true_i[-1]
            pred_total += pred_i[-1]

        avg_loss = loss_total / float(len(batches))
        precision = crr / pred_total if pred_total > 0 else 0.
        recall = crr / true_total if true_total > 0 else 0.
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.

        write('\n\tTime: %f seconds' % (time.time() - start))
        write('\tAverage Negative Log Likelihood: %f' % avg_loss)
        write('\tLabel: F1:%f\tP:%f(%d/%d)\tR:%f(%d/%d)' % (f1, precision, crr, pred_total, recall, crr, true_total))

    def validate_label(self, batches):
        crr = 0.
        true_total = 0.
        pred_total = 0.

        start = time.time()
        for index, batch in enumerate(batches):
            if (index + 1) % 100 == 0:
                print '%d' % (index + 1),
                sys.stdout.flush()

            crr_i, true_i, pred_i = self.predict_label_func(*batch)
            crr += crr_i[-1]
            true_total += true_i[-1]
            pred_total += pred_i[-1]

        precision = crr / pred_total if pred_total > 0 else 0.
        recall = crr / true_total if true_total > 0 else 0.
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.

        write('\n\tTime: %f seconds' % (time.time() - start))
        write('\tLabel: F1:%f\tP:%f(%d/%d)\tR:%f(%d/%d)' % (f1, precision, crr, pred_total, recall, crr, true_total))
        return f1

    def predict_shift(self, batches):
        labels = []
        start = time.time()
        for index, batch in enumerate(batches):
            if (index + 1) % 100 == 0:
                print '%d' % (index + 1),
                sys.stdout.flush()

            if len(batch) == 0:
                label_pred = []
            else:
                label_pred = self.predict_shift_func(*batch)
            labels.append(label_pred)
        write('\n\tTime: %f seconds' % (time.time() - start))
        return labels

    def predict_label(self, batches):
        y = []
        start = time.time()
        for index, batch in enumerate(batches):
            if (index + 1) % 100 == 0:
                print '%d' % (index + 1),
                sys.stdout.flush()

            if len(batch[0][0]) < 2 or np.sum(batch[-1]) < 1:
                y_i = [[0]]
            else:
                y_i = self.predict_label_func(*batch)
            y.append(y_i)

        write('\n\tTime: %f seconds' % (time.time() - start))
        return y

    def predict_shift_and_label(self, samples):
        start = time.time()

        for index, sent in enumerate(samples):
            if (index + 1) % 100 == 0:
                print '%d' % (index + 1),
                sys.stdout.flush()

            if sent.n_words < 2:
                sent.results = [[0]]
            else:
                inputs = []
                if sent.word_ids_corpus is not None:
                    inputs.append([sent.word_ids_corpus])
                if sent.word_ids_emb is not None:
                    inputs.append([sent.word_ids_emb])

                shifts, labels = self.predict_shift_and_label_func(*inputs)
                sent.prd_indices = [i for i, shift in enumerate(shifts.flatten()) if shift > 0]
                sent.n_prds = len(sent.prd_indices)
                sent.results = labels

        write('\n\tTime: %f seconds' % (time.time() - start))
        return samples

    def predict_online_shift(self, sample):
        stack_a, stack_p, shift_proba = self.predict_shift_func(*sample)
        return stack_a, stack_p, shift_proba

    def predict_online_shift_and_label(self, sample):
        stack_a, stack_p, shift_proba, label_proba, label_pred = self.predict_shift_and_label_func(*sample)
        return stack_a, stack_p, shift_proba, label_proba, label_pred

    def predict_online(self, sample):
        if len(sample[0][0]) < 2:
            prd_indices = []
            labels = [[0]]
        else:
            shifts, labels = self.predict_shift_and_label_func(*sample)
            prd_indices = self._convert_shifts_to_prd_indices(shifts)
            labels = self._convert_labels_to_prd_labels(labels, prd_indices)
        return prd_indices, labels

    @staticmethod
    def _convert_shifts_to_prd_indices(shifts):
        return [i for i, shift in enumerate(shifts.flatten()) if shift > 0]

    @staticmethod
    def _convert_labels_to_prd_labels(labels, prd_indices):
        """
        :param labels: 1D: time_steps, 2D: n_words * n_words; elem=label id
        :param prd_indices: 1D: n_prds; elem=word index
        :return: 1D: n_prds, 2D: time_steps, 3D: n_words(arg); elem=label id
        """
        n_words = len(labels)
        # 1D: time_steps, 2D: n_words(prd), 3D: n_words(arg); elem=label id
        labels = np.reshape(labels, (n_words, n_words, n_words))
        return [labels[:, p_index] for p_index in prd_indices]
