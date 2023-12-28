import math
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod

_key = jax.random.PRNGKey(42)


class Module(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def xi(self):
        pass

    @abstractmethod
    def theta(self):
        pass

    @abstractmethod
    def apply_xi_grad(self, lr):
        pass

    @abstractmethod
    def apply_theta_grad(self, lr):
        pass

    @abstractmethod
    def energy(self):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, x):
        pass

def energy_fn(m: Module, xi, theta):
    return jnp.average(m.energy(), axis=0)


class Network():

    def __init__(self, m: Module):
        self._m = m

    def predict(self, x):
        return self._m.predict(x)

    def energy(self):
        return energy_fn(self._m, self._m.xi(), self._m.theta())

    def learn(self, xi, xo, il_lr=0.1, theta_lr=0.01, eplison=1e-3, T=100):
        energy = jnp.average(self._m.energy())
        start_energy = energy
        prev_energy = energy
        for i, _ in enumerate(range(T)):
            print([x.shape for x in xi])
            xi_grad = jax.grad(energy_fn, argnums=1)(self._m, self._m.xi(), self._m.theta())
            print([g.shape for g in xi_grad])
            self._m.apply_xi_grad(xi_grad, il_lr)
            energy = jnp.average(self._m.energy(), axis=0)
            delta = prev_energy - energy
            print(delta)
            prev_energy = energy
            if delta / energy < eplison:
                print(f'IL energy: [{start_energy:.3e}, {energy:.3e}]')
                break
            if i == T - 1:
                print(f'IL energy: [{start_energy:.3e}, {prev_energy:.3e}] [iteration ended]')
        # learn theta
        theta_grad = jax.grad(energy_fn, argnums=2)(self._m, self._m.xi(), self._m.theta())
        self._m.apply_theta_grad(theta_grad, theta_lr)
        print(f'Theta energy: [{prev_energy:.3e}, {energy:.3e}]')


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

    def xi(self):
        return self._xi

    def theta(self):
        return self._theta

    def apply_xi_grad(self, xi_grad, lr):
        self._xi -= lr * xi_grad

    def apply_theta_grad(self, theta_grad, lr):
        self._theta -= lr * theta_grad

    def energy(self):
        if self._xi is None or self._mu is None or self._eo is None:
            raise ('must call forward and backward first')
        return 0.5 * jnp.sum(self._eo ** 2, axis=1)

    def predict(self, x):
        return jnp.einsum('io,bi->bo', self._theta, self._f(x))

    def forward(self, xi):
        if self._xi is None:
            self._xi = xi
        self._mu = jnp.einsum('io,bi->bo', self._theta, self._f(self._xi))
        return self._mu

    def backward(self, xo):
        if self._xi is None or self._mu is None:
            raise ('must call forward first')
        self._eo = xo - self._mu
        return self._xi


class Sequential(Module):

    def __init__(self, layers: list[Module]):
        super(Sequential, self).__init__('Sequential')
        self._layers = layers
        for idx, _ in enumerate(self._layers):
            if idx != 0:
                self._layers[idx].set_prev(self._layers[idx-1])

    def xi(self):
        return [l.xi() for l in self._layers]

    def theta(self):
        return [l.theta() for l in self._layers]

    def apply_xi_grad(self, xi_grad, lr):
        for i, _ in enumerate(self._layers):
            print(self._layers[i]._xi.shape)
            print(xi_grad[i].shape)
            self._layers[i].apply_xi_grad(xi_grad[i], lr)

    def apply_theta_grad(self, theta_grad, lr):
        for i, _ in enumerate(self._layers):
            self._layers[i].apply_theta_grad(theta_grad[i], lr)

    def energy(self):
        e = 0
        for l in self._layers:
            e += l.energy()
        return e

    def predict(self, x):
        for l in self._layers:
            x = l.predict(x)
        return x

    def forward(self, x):
        for l in self._layers:
            x = l.forward(x)
        return x

    def backward(self, x):
        for l in reversed(self._layers):
            x = l.backward(x)
        return x
