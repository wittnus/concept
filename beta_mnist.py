import tensorflow_datasets as tfds
import chex
import jax
import jax.numpy as jnp
import numpy as np
from chex import Array
from jax.tree_util import tree_map
import operator
from functools import partial
from einshape import jax_einshape as einshape
from jax import random
from jax.random import PRNGKey

def fuzz_pixels(key, image):
    noise = random.uniform(key, image.shape)
    return image.astype(jnp.float32) + noise

@chex.dataclass
class Datum:
    image: Array
    label: Array

    @classmethod
    def init(cls, batch, key) -> 'Datum':
        images, labels = batch['image'], batch['label']
        images = fuzz_pixels(key, images + 1.)
        images = images.astype(jnp.float32) / 258.0
        return cls(image=images, label=labels)

    @property
    def pos(self) -> Array:
        return self.image

    @property
    def neg(self) -> Array:
        return 1 - self.image

    def filter(self, classes: Array) -> 'Datum':
        mask = jnp.isin(self.label, classes)
        return tree_map(operator.itemgetter(mask), self)

    def resize(self, height: int, width: int) -> 'Datum':
        new_shape = self.image.shape[:-3] + (height, width, self.image.shape[-1])
        new_image = jax.image.resize(self.image, new_shape, method='linear')
        return self.replace(image=new_image)

    def flatten(self) -> 'Datum':
        reshape = partial(einshape, "...hwc->...(hwc)")
        return self.replace(image=reshape(self.image))

    def shuffle(self, key: PRNGKey) -> 'Datum':
        perm = random.permutation(key, self.image.shape[0])
        return tree_map(operator.itemgetter(perm), self)

    def pairs(self, key: PRNGKey, widthwise=True) -> 'Datum':
        other = self.shuffle(key)
        axis = -2 if widthwise else -3
        merged_image = jnp.concatenate([self.image, other.image], axis=axis)
        H, W = self.image.shape[-3], self.image.shape[-2]
        resized = self.replace(image=merged_image).resize(H, W)
        return resized


def _load_mnist():
    key = PRNGKey(0)
    mnist = tfds.load('mnist', split='train', batch_size=-1)
    mnist = tfds.as_numpy(mnist)
    return Datum.init(mnist, key)

def load_mnist(classes: Array=jnp.arange(10), height: int=28, width: int=28) -> Datum:
    mnist = _load_mnist()
    mnist = mnist.filter(classes)
    mnist = mnist.pairs(PRNGKey(0), widthwise=True)
    #mnist = mnist.pairs(PRNGKey(1), widthwise=False)
    mnist = mnist.resize(height, width)
    mnist = mnist.flatten()
    return mnist

def main():
    mnist = _load_mnist()
    mnist = mnist.resize(16, 16)
    classes = jnp.array([0])
    mnist = mnist.filter(classes)
    mnist = mnist.flatten()
    mnist0 = tree_map(operator.itemgetter(0), mnist)
    print(mnist0.image.shape)
    print(jnp.max(mnist.image))
    print(jnp.min(mnist.image))
    assert (mnist.pos >= 0).all()
    assert (mnist.pos <= 1).all()
    assert (mnist.neg >= 0).all()
    assert (mnist.neg <= 1).all()
    assert (1.0 - mnist.pos - mnist.neg <= 1e-6).all()

if __name__ == '__main__':
    main()



