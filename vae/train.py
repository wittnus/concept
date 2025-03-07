# Copyright 2023 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training and evaluation logic."""

from absl import logging
from flax import linen as nn
from flax.training import orbax_utils
import input_pipeline
import supervised_mnist
import models
import utils as vae_utils
from flax.training import train_state
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import optax
import tensorflow_datasets as tfds
import numpy as np

import orbax.checkpoint
from tqdm import trange, tqdm
from pathlib import Path


@jax.vmap
def kl_divergence(mean, logvar):
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
  logits = nn.log_sigmoid(logits)
  return -jnp.sum(
      labels * logits + (1.0 - labels) * jnp.log(-jnp.expm1(logits))
  )

@jax.vmap
def cross_entropy_with_logits(logits, labels):
    labels = labels.astype(jnp.int32)
    lprobs = nn.log_softmax(logits).take(labels, axis=-1)
    return -lprobs
    return -jnp.sum(labels * nn.log_softmax(logits), axis=-1)


def compute_metrics(recon_x, x, mean, logvar):
  bce_loss = binary_cross_entropy_with_logits(recon_x, x).mean()
  kld_loss = kl_divergence(mean, logvar).mean()
  return {'bce': bce_loss, 'kld': kld_loss, 'loss': bce_loss + kld_loss}

def compute_super_metrics(digits_logits, colors_logits, digits, colors):
  digit_loss = cross_entropy_with_logits(digits_logits, digits).mean()
  color_loss = cross_entropy_with_logits(colors_logits, colors).mean()
  return {'digit_loss': digit_loss, 'color_loss': color_loss}


def train_step(state, batch, z_rng, latents):
  def loss_fn(params):
    recon_x, mean, logvar, digit_logits, color_logits = models.model(latents).apply(
        {'params': params}, batch.image, z_rng
    )

    bce_loss = binary_cross_entropy_with_logits(recon_x, batch.image).mean()
    kld_loss = kl_divergence(mean, logvar).mean()
    digit_loss = cross_entropy_with_logits(digit_logits, batch.digit).mean() * 0e1
    color_loss = cross_entropy_with_logits(color_logits, batch.color).mean() * 0e1
    loss = bce_loss + kld_loss + digit_loss + color_loss
    return loss

  grads = jax.grad(loss_fn)(state.params)
  return state.apply_gradients(grads=grads)


def eval_f(params, data, z, z_rng, latents):
  images = data.image
  def eval_model(vae):
    recon_images, mean, logvar, digit_logits, color_logits = vae(images, z_rng)
    comparison = jnp.concatenate([
        images[:8].reshape(-1, 28, 28, 1),
        recon_images[:8].reshape(-1, 28, 28, 1),
    ])

    generate_images = vae.generate(z)
    generate_images = generate_images.reshape(-1, 28, 28, 1)
    metrics = compute_metrics(recon_images, images, mean, logvar)
    super_metrics = compute_super_metrics(digit_logits, color_logits, data.digit, data.color)
    metrics.update(super_metrics)
    metrics['loss'] += super_metrics['digit_loss'] + super_metrics['color_loss']
    return metrics, comparison, generate_images

  return nn.apply(eval_model, models.model(latents))({'params': params})


def save_model(params, path):
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    checkpointer.save(path, params, save_args=save_args, force=True)

def save_encodings(latents, state, ds, path):
    vae = models.model(latents)
    def compute(batch):
        model = vae.bind({'params': state.params})
        mean, logvar = model.encoder(batch.image)
        recon = model.generate(mean)
        return recon, mean

    recon = []
    encoding = []
    for batch in ds.batched(256):
        recon_batch, mean = compute(batch)
        recon.append(recon_batch)
        encoding.append(mean)
    recon = jnp.concatenate(recon)
    encoding = jnp.concatenate(encoding)
    np.savez(path, recon=recon, encoding=encoding)

def save_encoded_ds(latents, state, ds, path):
    vae = models.model(latents)
    def compute(batch):
        model = vae.bind({'params': state.params})
        mean, logvar = model.encoder(batch.image)
        digit_logit = nn.log_softmax(model.classify_digit(mean), axis=-1)
        color_logit = nn.log_softmax(model.classify_color(mean), axis=-1)
        recon = model.generate(mean)
        return recon, mean, logvar, digit_logit, color_logit
    recon, encoding, logvar, digit_logit, color_logit = compute(ds)
    image, digit, color = ds.image, ds.digit, ds.color
    np.savez(path, recon=recon, encoding=encoding, logvar=logvar, digit_logit=digit_logit, color_logit=color_logit, image=image, digit=digit, color=color)




def train_and_evaluate(config: ml_collections.ConfigDict):
  """Train and evaulate pipeline."""
  rng = random.key(0)
  rng, key = random.split(rng)

  #ds_builder = tfds.builder('binarized_mnist')
  #ds_builder.download_and_prepare()

  logging.info('Initializing dataset.')
  #train_ds = input_pipeline.build_train_set(config.batch_size, ds_builder)
  #test_ds = input_pipeline.build_test_set(ds_builder)
  allowed_digits = np.array([0, 1, 2, 3])
  full_train_ds = supervised_mnist.load_dataset(split='train', allowed_digits=allowed_digits).flatten()
  train_ds = full_train_ds.batch_stream(config.batch_size, key=random.key(0))
  test_ds = supervised_mnist.load_dataset(split='test', allowed_digits=allowed_digits).flatten()

  logging.info('Initializing model.')
  init_data = jnp.ones((config.batch_size, 784), jnp.float32)
  params = models.model(config.latents).init(key, init_data, rng)['params']

  state = train_state.TrainState.create(
      apply_fn=models.model(config.latents).apply,
      params=params,
      tx=optax.adam(config.learning_rate),
  )

  rng, z_key, eval_rng = random.split(rng, 3)
  z = random.normal(z_key, (64, config.latents))

  steps_per_epoch = len(full_train_ds.image) // config.batch_size
  #steps_per_epoch = (
  #    ds_builder.info.splits['train'].num_examples // config.batch_size
  #)

  save_path = Path(__file__).parent / 'results'

  for epoch in range(config.num_epochs):
    for _ in trange(steps_per_epoch):
      batch = next(train_ds)
      rng, key = random.split(rng)
      state = train_step(state, batch, key, config.latents)

    metrics, comparison, sample = eval_f(
        state.params, test_ds, z, eval_rng, config.latents
    )
    vae_utils.save_image(
            comparison, save_path / f'reconstruction_{epoch:02d}.png', nrow=8
    )
    vae_utils.save_image(sample, save_path / f'sample_{epoch:02d}.png', nrow=8)

    print(f"eval epoch: {epoch + 1}, loss: {metrics['loss']:.4f}, BCE: {metrics['bce']:.4f}, KLD: {metrics['kld']:.4f}, digit: {metrics['digit_loss']:.4f}, color: {metrics['color_loss']:.4f}")
    save_encodings(config.latents, state, test_ds, save_path / 'encodings.npz')
    save_encoded_ds(config.latents, state, test_ds, save_path / 'encoded_ds.npz')
    save_model(state.params, save_path / f'single_save')
