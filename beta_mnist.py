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
from pathlib import Path
import orbax.checkpoint
import vae.models as models
from einshape import jax_einshape as einshape

def load_vae_model(latents: int=20, shape=(784,), path=None):
    if path is None:
        path = Path(__file__).parent / Path("vae/results/single_save")
    key, rng = random.split(random.PRNGKey(0))
    init_data = jnp.ones((1,) + shape, dtype=jnp.float32)
    model = models.model(latents)
    params = model.init(key, init_data, rng)["params"]
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    params = checkpointer.restore(path, item=params)
    return model, params

def vae_encode(model, params, data):
    mean, logvar = model.bind(dict(params=params)).encoder(data)
    return mean, logvar

def vae_decode(model, params, z):
    return model.apply(dict(params=params), z, method=model.generate)

@chex.dataclass
class EncodedDatum:
    x: Array
    label: Array

def load_encoded_mnist(classes: Array=jnp.arange(10)):
    encoded_data = jnp.load("vae/results/encoded_ds.npz")
    data = encoded_data["encoding"]
    labels = encoded_data["digit"]
    keep = jnp.isin(labels, classes)
    return EncodedDatum(x=data[keep], label=labels[keep])


def fuzz_pixels(key, image):
    noise = random.uniform(key, image.shape)
    return image.astype(jnp.float32) + noise

@chex.dataclass
class Datum:
    x: Array
    label: Array

    @classmethod
    def init(cls, batch, key) -> 'Datum':
        images, labels = batch['image'], batch['label']
        images = fuzz_pixels(key, images + 1.)
        images = images.astype(jnp.float32) / 258.0
        return cls(x=images, label=labels)

    def filter(self, classes: Array) -> 'Datum':
        mask = jnp.isin(self.label, classes)
        return tree_map(operator.itemgetter(mask), self)

    def resize(self, height: int, width: int) -> 'Datum':
        new_shape = self.x.shape[:-3] + (height, width, self.x.shape[-1])
        new_image = jax.image.resize(self.x, new_shape, method='linear')
        return self.replace(x=new_image)

    def flatten(self) -> 'Datum':
        reshape = partial(einshape, "...hwc->...(hwc)")
        return self.replace(x=reshape(self.x))

    def shuffle(self, key: PRNGKey) -> 'Datum':
        perm = random.permutation(key, self.x.shape[0])
        return tree_map(operator.itemgetter(perm), self)

    def pairs(self, key: PRNGKey, widthwise=True) -> 'Datum':
        other = self.shuffle(key)
        axis = -2 if widthwise else -3
        merged_image = jnp.concatenate([self.x, other.x], axis=axis)
        H, W = self.x.shape[-3], self.x.shape[-2]
        resized = self.replace(x=merged_image).resize(H, W)
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
    print(mnist0.x.shape)
    print(jnp.max(mnist.x))
    print(jnp.min(mnist.x))

if __name__ == '__main__':
    main()



