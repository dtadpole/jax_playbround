import math
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod

_key = jax.random.PRNGKey(42)


class Module(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, x):
        pass

    @abstractmethod
    def theta_update(self, lr):
        pass


class Network():

    def __init__(self, m: Module):
        self._m = m

    def predict(self, x):
        return self._m.predict(x)

    def inference_learn(self, xi, xo, lr=0.5, T=10):
        for i in range(T):
            self._m.forward(xi)
            self._m.backward(xo, lr)

    def theta_update(self, lr=0.1):
        self._m.theta_update(lr)


class Dense(Module):

    def __init__(self, input: int, output: int, f=lambda x: x, prev: Module = None):
        super(Dense, self).__init__('Dense')
        self._input = input
        self._output = output
        self._f = f
        self._df = jax.grad(self._f)
        self._theta = jnp.zeros((input, output))
        self._initialize_params()
        self._xi = None
        self._mu = None
        self._eo = None
        self._prev = prev

    def set_prev(self, prev: Module):
        self._prev = prev

    def _initialize_params(self):
        # Xavier Initialization w/ normal distribution (instead of uniform)
        self._theta = jax.random.normal(_key, (self._input, self._output)) \
            * math.sqrt(1.0/self._input)

    def predict(self, x):
        return jnp.einsum('io,bi->bo', self._theta,
                          jax.vmap(self._f)(x))

    def forward(self, xi):
        if self._xi is None:
            self._xi = xi
        self._mu = jnp.einsum('io,bi->bo', self._theta,
                              jax.vmap(self._f)(self._xi))
        return self._mu

    def backward(self, xo, lr):
        if self._xi is None or self._mu is None:
            raise ('must call forward first')
        self._eo = xo - self._mu
        self._mu = None
        # check for prev layer
        # if self._prev is not None and self._prev._eo is None:
        #     raise ('must call backward on prev first')
        # calculate self._xi
        if self._prev is not None:
            if self._prev._eo is None:
                self._xi += lr * \
                    jax.vmap(jax.vmap(self._df))(self._xi) * \
                    jnp.einsum('io,bo->bi', self._theta, self._eo)
            else:
                self._xi += lr * (-self._prev._eo + jax.vmap(jax.vmap(self._df))(self._xi) *
                                  jnp.einsum('io,bo->bi', self._theta, self._eo))
        return self._xi

    def theta_update(self, lr):
        if self._xi is None:
            raise ('must call forward first')
        if self._eo is None:
            raise ('must call backward first')
        self._theta -= lr * \
            jnp.einsum('bo,bi->io', self._eo, jax.vmap(self._f)(self._xi))
        self._xi = None
        self._eo = None


class Sequential(Module):

    def __init__(self, layers: list[Module]):
        super(Sequential, self).__init__('Sequential')
        self._layers = layers
        for idx, _ in enumerate(self._layers):
            if idx != 0:
                self._layers[idx].set_prev(self._layers[idx-1])

    def predict(self, x):
        for l in self._layers:
            x = l.predict(x)
        return x

    def forward(self, x):
        for l in self._layers:
            x = l.forward(x)
        return x

    def backward(self, x, lr):
        for l in reversed(self._layers):
            x = l.backward(x, lr)
        return x

    def theta_update(self, lr):
        for l in reversed(self._layers):
            l.theta_update(lr)
