import sys
import time

import numpy as np
import theano
import theano.tensor as T

from ..utils import write
from ..model_api import ModelAPI
from models import ISRLSystem, ShiftModel, LabelModel


class ISRLSystemAPI(ModelAPI):
    def __init__(self, argv):
        super(ISRLSystemAPI, self).__init__(argv)
        self.predict_shift_func = None
        self.predict_label_func = None
        self.predict_shift_and_label_func = None

    def load_shift_model_params(self):
        params = self.load_data(self.argv.load_pi_param)
        assert len(self.model.shift_model.params) == len(params)
        for p1, p2 in zip(self.model.shift_model.params, params):
            p1.set_value(p2)

    def load_label_model_params(self):
        params = self.load_data(self.argv.load_lp_param)
        assert len(self.model.label_model.params) == len(params)
        for p1, p2 in zip(self.model.label_model.params, params):
            p1.set_value(p2)

    def set_model(self, **kwargs):
        pi_args = kwargs['pi_args']
        lp_args = kwargs['lp_args']

        self.vocab_word_corpus = kwargs['vocab_word_corpus']
        self.vocab_word_emb = kwargs['vocab_word_emb']
        self.vocab_label = kwargs['vocab_label']

        shift_model = ShiftModel()
        shift_model.compile(
            inputs=None,
            vocab_word_corpus_size=self.vocab_word_corpus.size() if self.vocab_word_corpus else 0,
            vocab_word_emb_size=self.vocab_word_emb.size() if self.vocab_word_emb else 0,
            input_dim=kwargs['init_emb'].shape[1] if kwargs['init_emb'] is not None else pi_args.emb_dim,
            hidden_dim=pi_args.hidden_dim,
            n_layers=pi_args.n_layers,
            rnn_unit=pi_args.rnn_unit,
            init_emb=kwargs['init_emb'],
            init_emb_fix=pi_args.init_emb_fix,
            drop_rate=0.0
        )

        label_model = LabelModel()
        label_model.compile(
            inputs=None,
            vocab_word_corpus_size=self.vocab_word_corpus.size() if self.vocab_word_corpus else 0,
            vocab_word_emb_size=self.vocab_word_emb.size() if self.vocab_word_emb else 0,
            input_dim=[kwargs['init_emb'].shape[1] if kwargs['init_emb'] is not None else lp_args.emb_dim, 5],
            hidden_dim=lp_args.hidden_dim,
            output_dim=kwargs['vocab_label'].size(),
            n_layers=lp_args.n_layers,
            rnn_unit=lp_args.rnn_unit,
            init_emb=kwargs['init_emb'],
            init_emb_fix=lp_args.init_emb_fix,
            drop_rate=0.0
        )

        self.model = ISRLSystem(inputs=self._set_inputs(),
                                shift_model=shift_model,
                                label_model=label_model)
        self._show_model_config()

    def _show_model_config(self):
        model = self.model
        write('\nModel configuration')

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

    def set_predict_func(self):
        write('\nBuilding a predict func...')

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

    def set_predict_given_gold_prds_func(self):
        write('\nBuilding a predict func...')
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

    def set_predict_online_func(self):
        write('\nBuilding a predict func...')

        if self.vocab_word_corpus and self.vocab_word_emb:
            x = T.imatrix('x')
        elif self.vocab_word_corpus:
            x = T.ivector('x_word_corpus')
        elif self.vocab_word_emb:
            x = T.ivector('x_word_emb')
        else:
            x = None

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
            inputs=[x, time_step],
            outputs=[stack_a[:, 0].flatten(),
                     T.sum(stack_p[:, 0], axis=0),
                     shift_proba,
                     label_proba,
                     label_pred],
            updates=updates,
            mode='FAST_RUN',
            on_unused_input='warn'
        )

    def predict(self, samples):
        start = time.time()

        for index, sent in enumerate(samples):
            if (index + 1) % 100 == 0:
                print '%d' % (index + 1),
                sys.stdout.flush()

            if sent.n_words < 2:
                sent.prd_props_sys = [['_']]
            else:
                inputs = []
                if sent.word_ids_corpus is not None:
                    inputs.append([sent.word_ids_corpus])
                if sent.word_ids_emb is not None:
                    inputs.append([sent.word_ids_emb])

                shifts, labels = self.predict_shift_and_label_func(*inputs)

                sent.prd_indices_sys = [i for i, shift in enumerate(shifts.flatten()) if shift > 0]
                sent.prd_props_sys = self._convert_labels_props(sent.prd_indices_sys, labels)

        write('\n\tTime: %f seconds' % (time.time() - start))
        return samples

    def _convert_labels_props(self, prd_indices, label_ids):
        """
        :param prd_indices: 1D: n_prds; elem=word index
        :param label_ids: 1D: time_steps, 2D: n_words(prd) * n_words(arg); elem=label id
        :return: 1D: n_prds, 2D: n_words, 3D: time_steps; elem=label id
        """
        n_words = len(label_ids)
        label_ids = np.reshape(label_ids, (n_words, n_words, n_words))

        labels = []
        for p_index in prd_indices:
            labels_i = []
            # 1D: time_steps, 2D: n_words(arg); elem;label id
            props = label_ids[:, p_index, :]
            for t, prop in enumerate(props):
                labels_i.append([self.vocab_label.get_word(label_id) for label_id in prop[:t + 1]])
            labels.append(labels_i)

        return labels

    def predict_given_gold_prds(self, batches):
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

    def predict_online(self, sample):
        stack_a, stack_p, shift_proba, label_proba, label_pred = self.predict_shift_and_label_func(*sample)
        return stack_a, stack_p, shift_proba, label_proba, label_pred
