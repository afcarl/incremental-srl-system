import theano.tensor as T

from core import Dense, Dropout
from recurrent import GRU, LSTM


class StackLayer(object):
    def __init__(self, name='StackLayer'):
        self.name = name
        self.layers = []
        self.params = []

    def _set_layers(self):
        raise NotImplementedError

    def _set_rnn_unit(self, unit_type):
        return GRU if unit_type == 'gru' else LSTM

    def _set_connect_unit(self, connect_type):
        return Dense

    def _set_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.params)
        return params

    def forward(self, x, **kwargs):
        raise NotImplementedError


class UniRNNLayer(StackLayer):
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_layers,
                 unit_type,
                 connect_type,
                 drop_rate=0.0):
        name = 'UniRNNLayer-%d:(%dx%d):%s:%s' % (n_layers, input_dim, output_dim, unit_type, connect_type)
        super(UniRNNLayer, self).__init__(name=name)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.rnn_unit = self._set_rnn_unit(unit_type=unit_type)
        self.connect_unit = self._set_connect_unit(connect_type=connect_type)
        self.dropout = Dropout(drop_rate)

        self.layers = self._set_layers()
        self.params = self._set_params()

    def _set_layers(self):
        layers = []
        for i in xrange(self.n_layers):
            layers.append(self.rnn_unit(input_dim=self.input_dim if i == 0 else self.output_dim,
                                        output_dim=self.output_dim))
            layers.append(self.connect_unit(input_dim=self.input_dim+self.output_dim if i == 0 else self.output_dim*2,
                                            output_dim=self.output_dim,
                                            activation='relu'))
        return layers

    def forward(self, x, mask=None, is_train=False):
        n_layers = len(self.layers) / 2
        for i in xrange(n_layers):
            if mask is None:
                h = self.layers[i * 2].forward(x=x)
                h = self.dropout.forward(x=h, is_train=is_train)
                x = self.layers[i * 2 + 1].forward(T.concatenate([x, h], axis=2))
            else:
                h = self.layers[i * 2].forward(x=x, mask=mask)
                h = self.dropout.forward(x=h, is_train=is_train)
                x = self.layers[i * 2 + 1].forward(T.concatenate([x, h], axis=2)) * mask
        return x


class BiRNNLayer(StackLayer):
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_layers,
                 unit_type,
                 drop_rate=0.0):
        name = 'BiRNNLayer-%d:(%dx%d):%s' % (n_layers, input_dim, output_dim, unit_type)
        super(BiRNNLayer, self).__init__(name=name)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.rnn_unit = self._set_rnn_unit(unit_type=unit_type)
        self.dropout = Dropout(drop_rate)

        self.layers = self._set_layers()
        self.params = self._set_params()

    def _set_layers(self):
        layers = []
        for i in xrange(self.n_layers):
            if i == 0:
                unit_f = self.rnn_unit(input_dim=self.input_dim, output_dim=self.output_dim)
                unit_b = self.rnn_unit(input_dim=self.input_dim, output_dim=self.output_dim)
            else:
                unit_f = self.rnn_unit(input_dim=2 * self.output_dim, output_dim=self.output_dim)
                unit_b = self.rnn_unit(input_dim=2 * self.output_dim, output_dim=self.output_dim)
            layers += [unit_f, unit_b]
        return layers

    def forward(self, x, mask=None, is_train=False):
        for i in xrange(self.n_layers):
            if mask is None:
                hf = self.layers[i * 2].forward(x=x)
                hb = self.layers[i * 2 + 1].forward(x=x[::-1])
            else:
                hf = self.layers[i * 2].forward(x=x, mask=mask)
                hb = self.layers[i * 2 + 1].forward(x=x[::-1], mask=mask[::-1])
            h = T.concatenate([hf, hb[::-1]], axis=2)
            x = self.dropout.forward(x=h, is_train=is_train)
        return x


class IntWeaveBiRNNLayer(StackLayer):
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_layers,
                 unit_type,
                 connect_type,
                 drop_rate=0.0):
        name = 'IntWeaveBiRNNLayer-%d:(%dx%d):%s:%s' % (n_layers, input_dim, output_dim, unit_type, connect_type)
        super(IntWeaveBiRNNLayer, self).__init__(name=name)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.rnn_unit = self._set_rnn_unit(unit_type=unit_type)
        self.connect_unit = self._set_connect_unit(connect_type=connect_type)
        self.dropout = Dropout(drop_rate)

        self.layers = self._set_layers()
        self.params = self._set_params()

    def _set_layers(self):
        layers = []
        for i in xrange(self.n_layers):
            layers.append(self.rnn_unit(input_dim=self.input_dim if i == 0 else self.output_dim,
                                        output_dim=self.output_dim))
            layers.append(self.connect_unit(input_dim=self.input_dim+self.output_dim if i == 0 else self.output_dim*2,
                                            output_dim=self.output_dim,
                                            activation='relu'))
        return layers

    def forward(self, x, mask=None, is_train=False):
        n_layers = len(self.layers) / 2
        for i in xrange(n_layers):
            if mask is None:
                h = self.layers[i * 2].forward(x=x)
                h = self.dropout.forward(x=h, is_train=is_train)
                x = self.layers[i * 2 + 1].forward(T.concatenate([x, h], axis=2))
            else:
                h = self.layers[i * 2].forward(x=x, mask=mask)
                h = self.dropout.forward(x=h, is_train=is_train)
                x = self.layers[i * 2 + 1].forward(T.concatenate([x, h], axis=2)) * mask
                mask = mask[::-1]
            x = x[::-1]
        if (n_layers % 2) == 1:
            return x[::-1]
        return x
