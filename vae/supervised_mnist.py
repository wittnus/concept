import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
import chex
from jax.tree_util import tree_map

@chex.dataclass
class Datum:
    image: chex.Array
    digit: chex.Array
    color: chex.Array

    def init(batch):
        images, labels = batch['image'], batch['label']
        return Datum(image=images, digit=labels, color=jnp.zeros_like(labels))

    def binarize(self, threshold=0.5):
        return self.replace(image=(self.image > threshold).astype(jnp.float32))

    def invert(self):
        color = jnp.ones_like(self.color) - self.color
        return self.replace(image=1 - self.image, color=color)

    def batch_stream(self, batch_size, key, axis=0):
        while True:
            key, subkey = jax.random.split(key)
            choice = jax.random.choice(subkey, jnp.arange(len(self.image)), (batch_size,), replace=False)
            yield tree_map(lambda x: x.take(choice, axis=axis), self)

    def batched(self, batch_size, axis=0):
        num_batches = len(self.image) // batch_size
        for i in range(num_batches):
            index = jnp.arange(i*batch_size, (i+1)*batch_size)
            yield tree_map(lambda x: x.take(index, axis=axis), self)

    def flatten(self):
        im_shape = self.image.shape
        pixels = int(jnp.prod(jnp.array(im_shape[-3:])))
        return self.replace(image=self.image.reshape(im_shape[:-3] + (pixels,)))

    def digit_filter(self, allowed_digits):
        mask = jnp.isin(self.digit, allowed_digits)
        return tree_map(lambda x: x[mask], self)

    def set_width(self, width):
        new_shape = self.image.shape[:-2] + (width,) + self.image.shape[-1:]
        new_image = jax.image.resize(self.image, new_shape, method='linear')
        return self.replace(image=new_image)

def side_by_side(d1, d2, margin):
    shape = d1.image.shape
    W = shape[-2]
    new_W = 2 * W - margin
    out = jnp.zeros(shape[:-2] + (new_W,) + shape[-1:])
    out = out.at[..., :W, :].set(d1.image)
    out = out.at[..., -W:, :].add(d2.image)
    return d1.replace(image=out)

def shuffle(data, key):
    perm = jax.random.permutation(key, len(data['image']))
    return tree_map(lambda x: x[perm], data)

def load_dataset(split, allowed_digits=None):
    ds = tfds.load('mnist', split=split, batch_size=-1)
    ds = tfds.as_numpy(ds)
    ds = Datum.init(ds)
    if allowed_digits is not None:
        ds = ds.digit_filter(allowed_digits)
    ds2 = shuffle(ds, jax.random.PRNGKey(1))
    ds = side_by_side(ds, ds2, margin=6)
    ds = ds.set_width(28)
    ds = ds.binarize()
    #ids = ds.invert()
    #ds = tree_map(lambda x, y: jnp.concatenate([x, y], axis=0), ds, ids)
    ds = shuffle(ds, jax.random.PRNGKey(0))
    return ds

def main():
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    ds_train = load_dataset('train')
    for key, value in ds_train.items():
        print(key, value.shape)
    stream = ds_train.batch_stream(32, jax.random.PRNGKey(0))
    batch = next(stream)
    print(tree_map(jnp.shape, batch))

if __name__ == '__main__':
    main()

