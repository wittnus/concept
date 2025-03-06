import models
import orbax.checkpoint
import jax
from jax.random import PRNGKey
from collections import namedtuple
from jax import numpy as jnp
from pathlib import Path

def load_model(config, path):
    key = PRNGKey(0)
    key, rng = jax.random.split(key)
    init_data = jnp.ones((config.batch_size, 784), jnp.float32)
    model = models.model(config.latents)
    params = models.model(config.latents).init(key, init_data, rng)["params"]
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    params = checkpointer.restore(path, item=params)
    return model, params

def encode(model, params, data):
    mean, logvar = model.bind(dict(params=params)).encoder(data)
    return mean, logvar

def decode(model, params, z):
    return model.apply(dict(params=params), z, method=model.generate)




def main():
    path = Path(__file__).parent.resolve() / Path("results/single_save")
    config = namedtuple("Config", ["batch_size", "latents"])(batch_size=128, latents=20)
    model, params = load_model(config, path)
    mean, logvar = encode(model, params, jnp.ones((config.batch_size, 784), jnp.float32))
    print(mean.shape)

if __name__ == "__main__":
    main()
