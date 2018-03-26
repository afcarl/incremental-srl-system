import numpy as np
import theano
import theano.tensor as T


def get_optimizer(argv):
    if argv.opt_type == 'adam':
        return Adam(lr=argv.lr,
                    grad_clip=argv.grad_clip)
    elif argv.opt_type == 'adagrad':
        return Adagrad(lr=argv.lr,
                       grad_clip=argv.grad_clip)
    elif argv.opt_type == 'adadelta':
        return Adadelta(lr=argv.lr,
                        grad_clip=argv.grad_clip)
    return SGD(lr=argv.lr,
               grad_clip=argv.grad_clip)


class Optimizer(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, grads, params):
        raise NotImplementedError

    @staticmethod
    def _grad_clipping(gradients, max_norm=5.0):
        global_grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), gradients)))
        multiplier = T.switch(global_grad_norm < max_norm, 1.0, max_norm / global_grad_norm)
        return [g * multiplier for g in gradients]


class SGD(Optimizer):
    def __init__(self, lr=0.01, grad_clip=False, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.lr = theano.shared(np.asarray(lr, dtype=theano.config.floatX), borrow=True)
        self.grad_clip = grad_clip

    def __call__(self, params, grads):
        updates = []
        if self.grad_clip:
            grads = self._grad_clipping(grads, max_norm=1.0)
        for p, g in zip(params, grads):
            updates.append((p, p - self.lr * g))
        return updates


class Adagrad(Optimizer):
    def __init__(self, lr=0.01, eps=1e-6, grad_clip=False, **kwargs):
        super(Adagrad, self).__init__(**kwargs)
        self.lr = theano.shared(np.asarray(lr, dtype=theano.config.floatX), borrow=True)
        self.eps = eps
        self.grad_clip = grad_clip

    def __call__(self, params, grads):
        updates = []
        if self.grad_clip:
            grads = self._grad_clipping(grads, max_norm=1.0)
        for p, g in zip(params, grads):
            v = p.get_value(borrow=True)
            acc = theano.shared(np.zeros(v.shape, dtype=v.dtype), broadcastable=p.broadcastable)
            acc_t = acc + g ** 2
            updates.append((acc, acc_t))
            updates.append((p, p - self.lr * g / T.sqrt(acc_t + self.eps)))
        return updates


class Adadelta(Optimizer):
    def __init__(self, lr=0.01, rho=0.95, eps=1e-6, grad_clip=False, **kwargs):
        super(Adadelta, self).__init__(**kwargs)
        self.lr = theano.shared(np.asarray(lr, dtype=theano.config.floatX), borrow=True)
        self.rho = rho
        self.eps = eps
        self.grad_clip = grad_clip

    def __call__(self, params, grads):
        updates = []

        if self.grad_clip:
            grads = self._grad_clipping(grads, max_norm=1.0)

        for p, g in zip(params, grads):
            v = p.get_value(borrow=True)
            acc = theano.shared(np.zeros(v.shape, dtype=v.dtype), broadcastable=p.broadcastable)
            delta_acc = theano.shared(np.zeros(v.shape, dtype=v.dtype), broadcastable=p.broadcastable)

            acc_t = self.rho * acc + (1 - self.rho) * g ** 2
            updates.append((acc, acc_t))

            update = (g * T.sqrt(delta_acc + self.eps) / T.sqrt(acc_t + self.eps))
            updates.append((p, p - self.lr * update))

            delta_acc_new = self.rho * delta_acc + (1 - self.rho) * update ** 2
            updates.append((delta_acc, delta_acc_new))

        return updates


class Adam(Optimizer):
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8, grad_clip=False, **kwargs):
        super(Adam, self).__init__(**kwargs)
        self.lr = theano.shared(np.asarray(lr, dtype=theano.config.floatX), borrow=True)
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.grad_clip = grad_clip

    def __call__(self, params, grads):
        updates = []
        i = theano.shared(np.asarray(.0, dtype=theano.config.floatX))
        i_t = i + 1.
        a_t = self.lr * T.sqrt(1 - self.b2 ** i_t) / (1 - self.b1 ** i_t)

        if self.grad_clip:
            grads = self._grad_clipping(grads, max_norm=1.0)

        for p, g in zip(params, grads):
            p_tm = p.get_value(borrow=True)
            v = theano.shared(np.zeros(p_tm.shape, dtype=p_tm.dtype), broadcastable=p.broadcastable)
            r = theano.shared(np.zeros(p_tm.shape, dtype=p_tm.dtype), broadcastable=p.broadcastable)

            v_t = self.b1 * v + (1. - self.b1) * g
            r_t = self.b2 * r + (1. - self.b2) * g ** 2

            step = a_t * v_t / (T.sqrt(r_t) + self.eps)

            updates.append((v, v_t))
            updates.append((r, r_t))
            updates.append((p, p - step))

        updates.append((i, i_t))
        return updates

