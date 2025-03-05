import models
import orbax.checkpoint
import jax
from jax.random import PRNGKey

def load_model(config, path):
    key = PRNGKey(0)
    key, rng = jax.random.split(key)
    init_data = jnp.ones((config.batch_size, 784), jnp.float32)
    model = models.model(config.latents)
    params = models.model(config.latents).init(key, init_data, rng)['params']
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    params = checkpointer.restore(path, item=params)
    return model, params

def encode(model, params, data):
    mean, logvar = model.bind(params).encoder(data)
    return mean, logvar

def decode(model, params, z):
    return model.apply(params, z, method=model.generate)

