# Copyright 2024 The Flax Authors.
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

"""VAE model definitions."""

from flax import linen as nn
from jax import random
import jax.numpy as jnp


class Encoder(nn.Module):
  """VAE Encoder."""

  latents: int

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(500, name='fc1')(x)
    x = nn.relu(x)
    mean_x = nn.Dense(self.latents, name='fc2_mean')(x)
    logvar_x = nn.Dense(self.latents, name='fc2_logvar')(x)
    return mean_x, logvar_x


class Decoder(nn.Module):
  """VAE Decoder."""

  @nn.compact
  def __call__(self, z):
    z = nn.Dense(500, name='fc1')(z)
    z = nn.relu(z)
    z = nn.Dense(784, name='fc2')(z)
    return z


class VAE(nn.Module):
  """Full VAE model."""

  latents: int = 20

  def setup(self):
    self.encoder = Encoder(self.latents)
    self.decoder = Decoder()
    self.classify_digit = nn.Dense(10)
    self.classify_color = nn.Dense(2)

  def __call__(self, x, z_rng):
    mean, logvar = self.encoder(x)
    z = reparameterize(z_rng, mean, logvar)
    digit_logits = self.classify_digit(z)
    color_logits = self.classify_color(z)
    recon_x = self.decoder(z)
    return recon_x, mean, logvar, digit_logits, color_logits

  def generate(self, z):
    return nn.sigmoid(self.decoder(z))


def reparameterize(rng, mean, logvar):
  std = jnp.exp(0.5 * logvar)
  eps = random.normal(rng, logvar.shape)
  return mean + eps * std


def model(latents):
  return VAE(latents=latents)
