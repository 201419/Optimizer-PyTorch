from keras import backend as K
from six.moves import zip
from keras.optimizers import Optimizer
from keras.legacy import interfaces

class OFRL(Optimizer):

    def __init__(self, lr=0.01, version=1., decay=0.,
                 schedule=None, m_rho=0.1, adagrad_epsilon=1e-08, **kwargs):
        super(OFRL, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.decay = K.variable(decay, name='decay')
            self.version = version
            self.schedule = schedule
            self.m_rho = m_rho
            self.adagrad_epsilon = adagrad_epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        shapes = [K.int_shape(p) for p in params]
        predictable = [K.zeros(shape) for shape in shapes]
        grad_sqr_accum = [K.zeros(shape) for shape in shapes]
        scheduler = [K.ones(shape) for shape in shapes]
        self.weights = [self.iterations] + predictable
        t = K.cast(self.iterations, K.floatx()) + 1
        for p, g, m, a, s in zip(params, grads, predictable, grad_sqr_accum, scheduler):
            # Update M
            if self.version == 1:
                new_m = g
            elif self.version == 2:
                new_m = (m * (t - 1) + g) / t
            elif self.version == 3:
                new_m = m * self.m_rho + (1-self.m_rho) * g
            else:
                raise ValueError('self.version {} is not recognized'.format(self.version))

            # Update the sum of squared gradient
            new_a = a + K.square(g)

            # Update learning rate schedule
            if self.schedule is None:
                new_s = s
            elif self.schedule == 'adagrad':
                new_s = s / (K.sqrt(new_a) + self.adagrad_epsilon)
                #new_s = s
            else:
                raise ValueError('self.schedule {} is not recognized'.format(self.schedule))

            # Update params
            new_p = self.update_param(p, g, lr, m, new_m, s, new_s)

            # Finally, apply the updates
            self.updates.append(K.update(m, new_m))
            self.updates.append(K.update(a, new_a))
            self.updates.append(K.update(s, new_s))

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def update_param(self, p, g, lr, m, new_m, s, new_s):
        return new_s / s  * p + lr * new_s * (m - new_m - g)

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'decay': float(K.get_value(self.decay)),
                  'schedule': self.schedule,
                  'version': self.version,
                  'm_rho': self.m_rho,
                  'adagrad_epsilon': self.adagrad_epsilon,
                  }
        base_config = super(OFRL, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class OMDA(OFRL):

    def update_param(self, p, g, lr, m, new_m, s, new_s):
        return p + lr * (s * m - s * g - new_s * new_m)


class optimAdam(Optimizer):
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., **kwargs):
        super(optimAdam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs
        ms_old = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs_old = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        for p, g, m, v, m_old, v_old in zip(params, grads, ms, vs, ms_old, vs_old):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            p_t = p - 2 * lr_t * m_t / (K.sqrt(v_t) + self.epsilon) + lr_t * m_old / (K.sqrt(v_old) + self.epsilon)

            self.updates.append(K.update(m_old, m))
            self.updates.append(K.update(v_old, v))
            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(optimAdam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class optimAdagrad(Optimizer):
    """Adagrad optimizer.
    It is recommended to leave the parameters of this optimizer
    at their default values.
    # Arguments
        lr: float >= 0. Learning rate.
        epsilon: float >= 0. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    """

    def __init__(self, lr=0.01, epsilon=None, decay=0., **kwargs):
        super(optimAdagrad, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr, name='lr')
            self.decay = K.variable(decay, name='decay')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        accumulators = [K.zeros(shape) for shape in shapes]
        self.weights = accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        for p, g, a in zip(params, grads, accumulators):
            new_a = a + K.square(g)  # update accumulator
            new_p = p - lr * g / (K.sqrt(new_a) + self.epsilon) + lr * g * K.clip(K.cast(self.iterations, K.floatx())-1, 0, 1) * (1 / (K.sqrt(a) + self.epsilon) - 1 / (K.sqrt(new_a) + self.epsilon))
            self.updates.append(K.update(a, new_a))

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))

        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(optimAdagrad, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
