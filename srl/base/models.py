import theano
import theano.tensor as T

from ..nn import Embedding, Dense, BiRNNLayer


class Model(object):
    def __init__(self):
        self.is_train = theano.shared(0, borrow=True)
        self.inputs = None
        self.input_layers = []
        self.hidden_layers = []
        self.output_layers = []
        self.layers = []
        self.params = []
        self.y_proba = None
        self.y_pred = None

    def compile(self, inputs, **kwargs):
        raise NotImplementedError

    def _set_params(self):
        for l in self.layers:
            self.params += l.params


class BaseModel(Model):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.feat_layer = None
        self.label_layer = None

    def compile(self, inputs, **kwargs):
        self.inputs = inputs
        self.feat_layer = FeatureLayer()
        self.feat_layer.compile(**kwargs)
        self.label_layer = SoftmaxLayer()
        self.label_layer.compile(**kwargs)
        self.layers = self.feat_layer.layers + self.label_layer.layers
        self._set_params()

    def calc_label_proba(self, inputs):
        """
        :param inputs: (x1, x2, ...); elem=Tensor Type
        :return: 1D: batch_size, 2D: n_words, 3D: n_labels; elem=proba
        """
        # 1D: n_words, 2D: batch_size, 3D: hidden_dim
        h = self.feat_layer.forward(inputs)
        # 1D: batch_size, 2D: n_words, 3D: n_labels
        o = self.label_layer.forward(h)
        return o

    @staticmethod
    def argmax_label_proba(label_proba):
        """
        :param label_proba: 1D: batch_size, 2D: n_words, 3D: n_labels; elem=proba
        :return: 1D: batch_size, 2D: n_words; elem=label id
        """
        return T.argmax(label_proba, axis=2)

    @staticmethod
    def calc_label_path_score(label_proba, y_true):
        """
        :param label_proba: 1D: batch_size, 2D: n_words, 3D: n_labels; elem=proba
        :param y_true: 1D: batch_size, 2D: n_words; elem=label id
        :return: sum of the log likelihoods
        """
        label_proba = label_proba.reshape((label_proba.shape[0] * label_proba.shape[1], -1))
        true_label_proba = label_proba[T.arange(y_true.flatten().shape[0]), y_true.flatten()].reshape(y_true.shape)
        return T.sum(T.log(true_label_proba), axis=1)


class FeatureLayer(Model):
    def compile(self, **kwargs):
        self._set_layers(kwargs)
        self._set_params()

    def forward(self, inputs, mask=None):
        embs = []
        for i in xrange(len(self.input_layers) - 1):
            # 1D: batch_size, 2D: n_words, 3D: input_dim
            emb_i = self.input_layers[i].forward(x=inputs[i], is_train=self.is_train)
            embs.append(emb_i)
        emb_mark = self.input_layers[-1].forward(x=inputs[-1], is_train=0)
        embs.append(emb_mark)

        # 1D: batch_size, 2D: n_words, 3D: input_dim
        x = T.concatenate(tensor_list=embs, axis=2)
        # 1D: n_words, 2D: batch_size, 3D: hidden_dim
        h = self.hidden_layers[0].forward(x=x.dimshuffle(1, 0, 2),
                                          mask=mask,
                                          is_train=self.is_train)
        return h

    def _set_layers(self, args):
        x_w_dim, x_m_dim = args['input_dim']
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

        emb_mark = Embedding(input_dim=2,
                             output_dim=x_m_dim,
                             init_emb=None,
                             param_init='xavier',
                             param_fix=0,
                             name='EmbMark')
        self.input_layers.append(emb_mark)

        #################
        # Hidden layers #
        #################
        hidden_layer = BiRNNLayer(input_dim=(len(self.input_layers) - 1) * x_w_dim + x_m_dim,
                                  output_dim=hidden_dim,
                                  n_layers=args['n_layers'],
                                  unit_type=args['rnn_unit'],
                                  drop_rate=drop_rate)
        self.hidden_layers = [hidden_layer]
        self.layers = self.input_layers + self.hidden_layers


class SoftmaxLayer(Model):
    def compile(self, **kwargs):
        self._set_layers(kwargs)
        self._set_params()

    def _set_layers(self, args):
        layer = Dense(input_dim=2 * args['hidden_dim'],
                      output_dim=args['output_dim'],
                      activation='softmax')
        self.layers = [layer]

    def forward(self, h):
        """
        :param h: 1D: n_words, 2D: batch_size, 3D: hidden_dim
        :return: 1D: batch_size, 2D: n_words, 3D: output_dim; elem=proba
        """
        return self.layers[0].forward(x=h.dimshuffle(1, 0, 2))


class SigmoidLayer(Model):
    def compile(self, **kwargs):
        self._set_layers(kwargs)
        self._set_params()

    def _set_layers(self, args):
        layer = Dense(input_dim=2 * args['hidden_dim'],
                      output_dim=1,
                      activation='sigmoid')
        self.layers = [layer]

    def forward(self, h):
        """
        :param h: 1D: n_words, 2D: batch_size, 3D: hidden_dim
        :return: 1D: batch_size, 2D: n_words; elem=proba
        """
        return self.layers[0].forward(x=h)
