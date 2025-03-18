import jax
from jax import numpy as jnp
import numpy as np

import chex
from chex import Array
from typing import Tuple, Any
from jax import random
from functools import partial
from jaxopt import LBFGS

@chex.dataclass
class Probe:

    def init(self, key: Array, num_classes: int, num_hidden: int) -> 'Probe':
        raise NotImplementedError
    def logits(self, hidden: Array) -> Array:
        raise NotImplementedError
    def fit(self, hidden: Array, obs: Array) -> 'Probe':
        raise NotImplementedError
    def loss(self, hidden: Array, obs: Array) -> Array:
        return -jnp.sum(self.logits(hidden) * obs, axis=-1)
    def accuracy(self, hidden: Array, obs: Array) -> Array:
        return jnp.mean(jnp.argmax(self.logits(hidden), axis=-1) == jnp.argmax(obs, axis=-1))
    def batch_loss(self, hidden: Array, obs: Array) -> Array:
        return jnp.mean(self.loss(hidden, obs))
    def batch_accuracy(self, hidden: Array, obs: Array) -> Array:
        return jnp.mean(self.accuracy(hidden, obs))

@chex.dataclass
class LogisticProbe(Probe):
    params: Array # (num_classes, num_hidden)

    @classmethod
    def init(cls, key: Array, num_classes: int, num_hidden: int) -> 'LogisticProbe':
        params = random.normal(key, (num_classes, num_hidden))
        return cls(params=params)

    def logits(self, hidden: Array) -> Array:
        out = hidden @ self.params.T
        return jax.nn.log_softmax(out, axis=-1)

    def fit(self, hidden: Array, obs: Array) -> 'LogisticProbe':
        def loss_fn(probe) -> Array:
            return probe.batch_loss(hidden, obs)
        solver = LBFGS(loss_fn, verbose=False)
        probe, state = solver.run(self)
        return probe

def init_and_fit_logistic_probe(key: Array, hidden: Array, obs: Array) -> LogisticProbe:
    probe = LogisticProbe.init(key, obs.shape[-1], hidden.shape[-1])
    return fit_logistic_probe(probe, hidden, obs)

def fit_logistic_probe(probe: LogisticProbe, hidden: Array, obs: Array) -> LogisticProbe:

    def loss_fn(probe: LogisticProbe) -> Array:
        return probe.batch_loss(hidden, obs)
    solver = LBFGS(loss_fn, verbose=False)
    probe, state = solver.run(probe)
    return probe

@chex.dataclass
class CountProbe(Probe):
    params: Array # (num_classes, num_hidden)

    @classmethod
    def init(cls, key: Array, num_classes: int, num_hidden: int) -> 'ChoiceProbe':
        params = jnp.ones((num_classes, num_hidden)) / num_hidden * 2
        return cls(params=params)

    def fit(self, choices: Array, obs: Array) -> 'ChoiceProbe':
        counts = choices[..., None, :] * obs[...,:, None]
        return self.replace(params=self.params + jnp.sum(counts, axis=0))

    def logits(self, hidden: Array) -> Array:
        out = hidden @ jnp.log(self.params.T)
        return jax.nn.log_softmax(out, axis=-1)

def init_and_fit_count_probe(key: Array, hidden: Array, obs: Array) -> CountProbe:
    probe = CountProbe.init(key, obs.shape[-1], hidden.shape[-1])
    return fit_count_probe(probe, hidden, obs)

def fit_count_probe(probe: CountProbe, hidden: Array, obs: Array) -> CountProbe:
    return probe.fit(hidden, obs)

