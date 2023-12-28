import math
import time

import numpy.random as npr
from pc import Module, Network, Dense, Sequential

import jax
from jax import jit, grad
from jax.scipy.special import logsumexp
import jax.numpy as jnp
import datasets


def accuracy(net, batch):
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(net.predict(inputs), axis=1)
    return jnp.mean(predicted_class == target_class)


if __name__ == "__main__":
    num_epochs = 50
    batch_size = 8192

    train_images, train_labels, test_images, test_labels = datasets.mnist()
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def data_stream():
        rng = npr.RandomState(0)
        while True:
            print('==> permutate')
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]
    batches = data_stream()

    net = Network(Sequential([
        Dense(784, 1024, jax.nn.tanh),
        Dense(1024, 256, jax.nn.tanh),
        Dense(256, 10, jax.nn.tanh),
    ]))

    train_acc = accuracy(net, (train_images, train_labels))
    test_acc = accuracy(net, (test_images, test_labels))
    print(f"Training set accuracy {train_acc*100:0.2f}%")
    print(f"Test set accuracy {test_acc*100:0.2f}%")

    for epoch in range(num_epochs):
        start_time = time.time()
        for t in range(num_batches):
            xi, xo = next(batches)
            # print(xi.shape, xo.shape)
            net.inference_learn(xi, xo)
            net.theta_update()
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} in {epoch_time:0.2f} sec")

        train_acc = accuracy(net, (train_images, train_labels))
        test_acc = accuracy(net, (test_images, test_labels))
        print(f"Training set accuracy {train_acc*100:0.2f}%")
        print(f"Test set accuracy {test_acc*100:0.2f}%")
