import theano.tensor as T

from ..lp import Model, BaseModel, SigmoidLayer
from ..nn import Embedding, BiRNNLayer


class PIModel(BaseModel):
    def compile(self, inputs, **kwargs):
        self.inputs = inputs
        self.feat_layer = PIFeatureLayer()
        self.feat_layer.compile(**kwargs)
        self.label_layer = SigmoidLayer()
        self.label_layer.compile(**kwargs)
        self.layers = self.feat_layer.layers + self.label_layer.layers
        self._set_params()

    def calc_proba(self, inputs):
        """
        :param inputs: (x1, x2, ...); elem=Tensor Type
        :return: 1D: batch_size, 2D: n_words; elem=proba
        """
        # 1D: n_words, 2D: batch_size, 3D: hidden_dim
        h = self.feat_layer.forward(inputs)
        # 1D: n_words, 2D: batch_size, 3D: 1; elem=proba
        o = self.label_layer.forward(h)
        # 1D: batch_size, 2D: n_words
        o = o.dimshuffle(1, 0, 2).reshape((h.shape[0], h.shape[1]))
        return o

    @staticmethod
    def binary_prediction(prd_proba):
        """
        :param prd_proba: 1D: batch_size, 2D: n_words; elem=proba
        :return: 1D: batch_size, 2D: n_words; elem=label id
        """
        return T.gt(prd_proba, 0.5)

    @staticmethod
    def get_binary_loss(y_true, y_proba):
        """
        :param y_true: 1D: batch_size, 2D: n_words; elem=0/1
        :param y_proba: 1D: batch_size, 2D: n_words; elem=proba
        :return:
        """
        return T.mean(T.nnet.binary_crossentropy(output=y_proba.flatten(),
                                                 target=y_true.flatten()))

    @staticmethod
    def calc_correct_predictions(y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        eqs = T.eq(y_true, y_pred)
        crr = T.sum(eqs * T.gt(y_pred, 0))
        true_total = T.sum(y_true)
        pred_total = T.sum(y_pred)
        return crr, true_total, pred_total


class PIFeatureLayer(Model):
    def compile(self, **kwargs):
        self._set_layers(kwargs)
        self._set_params()

    def forward(self, inputs, mask=None):
        embs = []
        for i in xrange(len(self.input_layers)):
            # 1D: batch_size, 2D: n_words, 3D: input_dim
            emb_i = self.input_layers[i].forward(x=inputs[i], is_train=self.is_train)
            embs.append(emb_i)

        # 1D: batch_size, 2D: n_words, 3D: input_dim
        x = T.concatenate(tensor_list=embs, axis=2)
        # 1D: n_words, 2D: batch_size, 3D: hidden_dim
        h = self.hidden_layers[0].forward(x=x.dimshuffle(1, 0, 2),
                                          mask=mask,
                                          is_train=self.is_train)
        return h

    def _set_layers(self, args):
        x_w_dim = args['input_dim']
        hidden_dim = args['hidden_dim']
        drop_rate = args['drop_rate']

        ################
        # Input layers #
        ################
        if args['vocab_word_corpus_size'] > 0:
            emb_corpus = Embedding(input_dim=args['vocab_word_corpus_size'],
                                   output_dim=x_w_dim,
                                   param_init='xavier',
                                   param_fix=0,
                                   drop_rate=drop_rate,
                                   name='EmbCorpus')
            self.input_layers.append(emb_corpus)

        if args['vocab_word_emb_size'] > 0:
            emb_init = Embedding(input_dim=args['vocab_word_emb_size'],
                                 output_dim=x_w_dim,
                                 init_emb=args['init_emb'],
                                 param_fix=args['init_emb_fix'],
                                 drop_rate=drop_rate,
                                 name='EmbInit')
            self.input_layers.append(emb_init)

        #################
        # Hidden layers #
        #################
        hidden_layer = BiRNNLayer(input_dim=len(self.input_layers) * x_w_dim,
                                  output_dim=hidden_dim,
                                  n_layers=args['n_layers'],
                                  unit_type=args['rnn_unit'],
                                  drop_rate=drop_rate)
        self.hidden_layers = [hidden_layer]
        self.layers = self.input_layers + self.hidden_layers
