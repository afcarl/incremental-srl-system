import numpy as np
import theano
import theano.tensor as T

from ..models import LPModel, PIModel


class ISRLSystem(object):
    def __init__(self, inputs, shift_model, label_model):
        self.inputs = inputs
        self.shift_model = shift_model
        self.label_model = label_model

        self.stack_a = theano.shared(value=np.zeros((50, 1, len(inputs)), dtype='int32'),
                                     name='stack_a',
                                     borrow=True)
        self.stack_p = theano.shared(value=np.zeros((50, 1, 50), dtype='int32'),
                                     name='stack_p',
                                     borrow=True)

    @staticmethod
    def argmax_label_proba(label_proba):
        """
        :param label_proba: 1D: time_steps, 2D: batch_size * n_words(prd) * n_words(arg), 3D: n_labels; elem=proba
        :return: 1D: time_steps, 2D: batch_size * n_words(prd) * n_words(arg); elem=label id
        """
        return T.argmax(label_proba, axis=2)

    def _set_initial_stacks(self):
        x = self.inputs[0]
        batch_size = x.shape[0]
        n_words = x.shape[1]
        # 1D: n_words(arg), 2D: batch_size, 3D: n_feats; elem=feat id
        stack_a_0 = T.zeros(shape=(n_words, batch_size, len(self.inputs)), dtype='int32')
        # 1D: n_words(arg), 2D: batch_size, 3D: n_words(prd); elem=0/1
        stack_p_0 = T.zeros(shape=(n_words, batch_size, n_words), dtype='int32')
        return stack_a_0, stack_p_0

    def _set_inputs(self):
        """
        :return: 1D: n_words(arg), 2D: batch_size, 3D: n_inputs; elem=feat id
        """
        x = []
        for x_i in self.inputs:
            x.append(x_i.dimshuffle(1, 0, 'x'))
        return T.concatenate(tensor_list=x, axis=2)

    def get_oracle_stack(self, y_shift):
        """
        :param y_shift: 1D: batch_size, 2D: n_words(arg); elem=0/1
        :return:
            label_proba: 1D: n_words, 2D: batch_size * n_words(prd), 3D: n_words(arg), 4D: n_labels; elem=proba
            stack_a: 1D: n_words, 2D: n_words(arg), 3D: batch_size; elem=feat id
            stack_p: 1D: n_words, 2D: n_words(arg), 3D: batch_size, 4D: n_words(prd); elem=0/1
        """
        stack_a_0, stack_p_0 = self._set_initial_stacks()
        [stack_a, stack_p, _], _ = theano.scan(fn=self.shift_model.shift_oracle,
                                               sequences=[self._set_inputs(),
                                                          y_shift.dimshuffle(1, 0)],
                                               outputs_info=[stack_a_0,
                                                             stack_p_0,
                                                             0]
                                               )
        return stack_a, stack_p

    def get_shift_proba(self):
        """
        :return: 1D: n_words, 2D: batch_size; elem=proba
        """
        stack_a_0, stack_p_0 = self._set_initial_stacks()
        [shift_proba, _, _], _ = theano.scan(fn=self.shift_model.get_shift_proba,
                                             sequences=[self._set_inputs()],
                                             outputs_info=[None, stack_a_0, 0]
                                             )
        return shift_proba

    def update_stack_a(self, x_t, time_step):
        """
        :param x_t: 1D: batch_size, 2D: n_feats; elem=feat id
        :param time_step: elem=int
        """
        return T.inc_subtensor(self.stack_a[time_step], x_t)

    def update_stack_p(self, shift_id_t, time_step):
        """
        :param shift_id_t: 1D: batch_size; elem=0/1
        :param time_step: elem=int
        """
        return shift_id_t * T.inc_subtensor(self.stack_p[time_step, :, time_step], 1) + (1 - shift_id_t) * self.stack_p

    def get_shift_proba_online(self, stack_a, time_step):
        """
        :param stack_a: 1D: n_words(arg), 2D: batch_size, 3D: n_inputs; elem=feat id
        :param time_step: int32
        :return: 1D: batch_size; elem=proba
        """
        # 1D: n_words, 2D: batch_size, 3D: hidden_dim
        h = self.shift_model.feat_layer.forward(inputs=stack_a.dimshuffle(2, 1, 0))
        return self.shift_model.label_layer.forward(h[time_step]).flatten()[0]

    @staticmethod
    def get_shift_id(shift_proba):
        return T.gt(shift_proba, 0.5)

    def get_label_proba_with_oracle_shift(self, y_shift):
        """
        :param y_shift: 1D: batch_size, 2D: n_words(arg); elem=0/1
        :return:
            label_proba: 1D: time_steps, 2D: batch_size * n_words(prd), 3D: n_words(arg), 4D: n_labels; elem=proba
            stack_a: 1D: time_steps, 2D: n_words(arg), 3D: batch_size, 4D: n_inputs; elem=feat id
            stack_p: 1D: time_steps, 2D: n_words(arg), 3D: batch_size, 4D: n_words(prd); elem=0/1
        """
        stack_a_0, stack_p_0 = self._set_initial_stacks()
        [label_proba, stack_a, stack_p, _], _ = theano.scan(fn=self._label_proba_step_with_oracle_shift,
                                                            sequences=[self._set_inputs(),
                                                                       y_shift.dimshuffle(1, 0)],
                                                            outputs_info=[None,
                                                                          stack_a_0,
                                                                          stack_p_0,
                                                                          0]
                                                            )
        return label_proba, stack_a, stack_p

    def _label_proba_step_with_oracle_shift(self, x_t, y_shift_t, stack_a, stack_p, time_step):
        """
        :param stack_a: 1D: n_words(arg), 2D: batch_size, 3D: n_inputs; elem=feat id
        :param stack_p: 1D: n_words(arg), 2D: batch_size, 3D: n_words(prd); elem=0/1
        """
        stack_a, stack_p, _ = self.shift_model.shift_oracle(x_t=x_t,
                                                            shift_id_t=y_shift_t,
                                                            stack_a=stack_a,
                                                            stack_p=stack_p,
                                                            time_step=time_step,
                                                            )
        # 1D: batch_size * n_words(prd), 2D: n_words(arg), 3D: n_labels; elem=proba
        label_proba = self.label_model.get_label_proba(stack_a=stack_a,
                                                       stack_p=stack_p,
                                                       time_step=time_step)
        # 1D: batch_size * n_words(prd) * n_words(arg), 2D: n_labels; elem=proba
        label_proba = label_proba.reshape((label_proba.shape[0] * label_proba.shape[1], -1))
        return label_proba, stack_a, stack_p, time_step + 1

    def get_shift_and_label_proba(self):
        """
        :return:
            shift_proba: 1D: time_steps, 2D: batch_size; elem=proba
            label_proba: 1D: time_steps, 2D: batch_size * n_words(prd), 3D: n_words(arg), 4D: n_labels; elem=proba
            stack_a: 1D: time_steps, 2D: n_words(arg), 3D: batch_size, 4D: n_inputs; elem=feat id
            stack_p: 1D: time_steps, 2D: n_words(arg), 3D: batch_size, 4D: n_words(prd); elem=0/1
        """
        stack_a_0, stack_p_0 = self._set_initial_stacks()
        [shift_proba, label_proba, stack_a, stack_p, _], _ = theano.scan(fn=self._shift_and_label_proba_step,
                                                                         sequences=[self._set_inputs()],
                                                                         outputs_info=[None,
                                                                                       None,
                                                                                       stack_a_0,
                                                                                       stack_p_0,
                                                                                       0]
                                                                         )
        return shift_proba, label_proba, stack_a, stack_p

    def _shift_and_label_proba_step(self, x_t, stack_a, stack_p, time_step):
        """
        :param stack_a: 1D: n_words(arg), 2D: batch_size, 3D: n_inputs; elem=feat id
        :param stack_p: 1D: n_words(arg), 2D: batch_size, 3D: n_words(prd); elem=0/1
        """
        # 1D: batch_size; elem=proba
        shift_proba, _, _ = self.shift_model.get_shift_proba(x_t=x_t,
                                                             stack_a=stack_a,
                                                             time_step=time_step)
        shift_id_t = T.gt(shift_proba, 0.5)
        stack_a, stack_p = self.shift_model.update_stack(x_t=x_t,
                                                         shift_id_t=shift_id_t,
                                                         stack_a=stack_a,
                                                         stack_p=stack_p,
                                                         time_step=time_step)
        # 1D: batch_size * n_words(prd), 2D: n_words(arg), 3D: n_labels; elem=proba
        label_proba = self.label_model.get_label_proba(stack_a=stack_a,
                                                       stack_p=stack_p,
                                                       time_step=time_step)
        label_proba = label_proba.reshape((label_proba.shape[0] * label_proba.shape[1], -1))
        return shift_proba, label_proba, stack_a, stack_p, time_step + 1


class ShiftModel(PIModel):
    def shift_oracle(self, x_t, shift_id_t, stack_a, stack_p, time_step):
        stack_a, stack_p = self.update_stack(x_t=x_t,
                                             shift_id_t=shift_id_t,
                                             stack_a=stack_a,
                                             stack_p=stack_p,
                                             time_step=time_step)
        return stack_a, stack_p, time_step + 1

    @staticmethod
    def update_stack(x_t, shift_id_t, stack_a, stack_p, time_step):
        """
        :param x_t: 1D: batch_size; elem=feat id
        :param shift_id_t: 1D: batch_size; elem=0/1
        :param stack_a: 1D: n_words(arg), 2D: batch_size, 3D: n_inputs; elem=feat id
        :param stack_p: 1D: n_words(arg), 2D: batch_size, 3D: n_words(prd); elem=0/1
        :param time_step: int32
        """
        shift_id_t = shift_id_t.dimshuffle('x', 0, 'x')
        stack_a = T.inc_subtensor(stack_a[time_step], x_t)
        stack_p = shift_id_t * T.inc_subtensor(stack_p[time_step, :, time_step], 1) + (1 - shift_id_t) * stack_p
        return stack_a, stack_p

    def get_shift_proba(self, x_t, stack_a, time_step):
        """
        :param x_t: 1D: batch_size; elem=feat id
        :param stack_a: 1D: n_words(arg), 2D: batch_size, 3D: n_inputs; elem=feat id
        :param time_step: int32
        :return: 1D: batch_size; elem=proba
        """
        stack_a = T.inc_subtensor(stack_a[time_step], x_t)
        # 1D: n_words, 2D: batch_size, 3D: hidden_dim
        h = self.feat_layer.forward(inputs=stack_a.dimshuffle(2, 1, 0))
        # 1D: batch_size; elem=proba
        label_proba = self.label_layer.forward(h[time_step]).flatten()

        return label_proba, stack_a, time_step + 1


class LabelModel(LPModel):
    def get_label_proba(self, stack_a, stack_p, time_step):
        """
        :param stack_a: 1D: n_words(arg), 2D: batch_size, 3D: n_inputs; elem=feat id
        :param stack_p: 1D: n_words(arg), 2D: batch_size, 3D: n_words(prd); elem=0/1
        :param time_step: int32
        :return: 1D: batch_size * n_words(prd), 2D: n_words(arg), 3D: n_labels; elem=proba
        """
        # 1D: n_inputs+1, 2D: batch_size * n_words(prd), 3D: n_words(arg); elem=proba
        stacks = self._concat_stacks(stack_a, stack_p)

        # 1D: batch_size * n_words(prd), 2D: n_words(arg)
        stack_p_reshaped = stack_p.dimshuffle(1, 2, 0).reshape((stack_p.shape[1] * stack_p.shape[2], stack_p.shape[0]))

        # 1D: n_words(arg); elem=0/1
        mask_for_rnn = self._make_mask_for_rnn(stack_p=stack_p_reshaped,
                                               time_step=time_step)

        # 1D: n_words(arg), 2D: batch_size * n_words(prd), 3D: hidden_dim
        h = self.feat_layer.forward(inputs=stacks,
                                    mask=mask_for_rnn)
        # 1D: batch_size * n_words(prd), 2D: n_words(arg), 3D: n_labels; elem=proba
        label_proba = self.label_layer.forward(h)

        # 1D: batch_size * n_words(prd), 2D: n_words(arg)
        mask_for_proba = self._make_mask_for_proba(stack_p=stack_p_reshaped,
                                                   time_step=time_step)

        return label_proba * mask_for_proba

    @staticmethod
    def _concat_stacks(stack_a, stack_p):
        """
        :param stack_a: 1D: n_words(arg), 2D: batch_size, 3D: n_inputs; elem=feat id
        :param stack_p: 1D: n_words(arg), 2D: batch_size, 3D: n_words(prd); elem=0/1
        :return: 1D: n_inputs+1, 2D: batch_size, 3D: n_words(arg); elem=proba
        """
        n_words_a = stack_a.shape[0]
        n_words_p = stack_p.shape[2]
        batch_size = stack_a.shape[1]

        # 1D: n_inputs, 2D: batch_size, 3D: n_words(arg)
        stack_a = stack_a.dimshuffle(2, 1, 0)
        # 1D: n_inputs, 2D: batch_size * n_words(prd), 3D: n_words(arg)
        stack_a = T.repeat(stack_a, repeats=n_words_p, axis=1)

        # 1D: batch_size, 2D: n_words(prd), 3D: n_words(arg)
        stack_p = stack_p.dimshuffle(1, 2, 0)
        # 1D: batch_size * n_words(prd), 2D: n_words(arg)
        stack_p = stack_p.reshape((1, batch_size * n_words_p, n_words_a))

        return T.concatenate([stack_a, stack_p], axis=0)

    @staticmethod
    def _make_mask_for_rnn(stack_p, time_step):
        """
        :param stack_p: 1D: batch_size * n_words(prd), 2D: n_words(arg); elem=0/1
        :return: 1D: n_words(arg); elem=0/1
        """
        zero = T.zeros((stack_p.shape[1], 1), dtype=theano.config.floatX)
        return T.inc_subtensor(zero[:time_step + 1], 1.)

    @staticmethod
    def _make_mask_for_proba(stack_p, time_step):
        """
        :param stack_p: 1D: batch_size * n_words(prd), 2D: n_words(arg); elem=0/1
        :return: 1D: batch_size * n_words(prd), 2D: n_words(arg), 3D: 1; elem=0/1
        """
        # 1D: batch_size * n_words(prd), 2D: n_words(arg)
        zero = T.zeros(stack_p.shape, dtype=theano.config.floatX)
        # 1D: 1D: batch_size * n_words(prd), 2D: n_words(arg)
        mask_a = T.inc_subtensor(zero[:, :time_step + 1], 1.)
        # 1D: batch_size * n_words(prd), 2D: 1
        mask_p = T.sum(stack_p, axis=1, keepdims=True)
        return (mask_a * mask_p).dimshuffle(0, 1, 'x')
