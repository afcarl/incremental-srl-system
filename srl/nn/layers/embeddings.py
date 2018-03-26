import theano

from core import Unit, Dropout


class Embedding(Unit):
    def __init__(self,
                 input_dim,
                 output_dim,
                 init_emb=None,
                 param_init='xavier',
                 param_fix=False,
                 drop_rate=0.0,
                 name=None):
        super(Embedding, self).__init__(name=name if name else 'Emb(%dx%d)' % (input_dim, output_dim))
        self.dropout = Dropout(drop_rate)

        self.W = self._set_weight(input_dim, output_dim, init_emb, param_init)
        if param_fix:
            self.params = []
        else:
            self.params = [self.W]

    def _set_weight(self, input_dim, output_dim, init_emb, param_init):
        if init_emb is None:
            return self._set_param(shape=(input_dim, output_dim),
                                   init_type=param_init,
                                   name='embedding')
        return theano.shared(init_emb)

    def forward(self, x, is_train=0):
        return self.dropout.forward(x=self.W[x], is_train=is_train)
