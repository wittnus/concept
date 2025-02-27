import numpy as np
import jax
from jax import numpy as jnp
import chex
from chex import Array
from jax.random import PRNGKey
from typing import Tuple


def reject(U: Array, v: Array) -> Array:
    """subtract projection of u onto v from u"""
    udotv = jnp.einsum("...ij,...i->...j", U, v)[..., None, :]
    vdotv = jnp.einsum("...i,...i->...", v, v)
    proj = udotv / vdotv
    proj = proj * v[..., None]
    return U - proj


@chex.dataclass
class DetNode:
    embedding: Array  # shape: (D, C)

    def create(D: int, C: int, key: PRNGKey) -> "DetNode":
        embedding = jax.random.normal(key, (D, C))
        M = embedding @ embedding.T
        invroot = jnp.linalg.inv(jnp.linalg.cholesky(M))
        embedding = invroot @ embedding
        return DetNode(embedding=embedding)

    @property
    def D(self) -> int:
        return self.embedding.shape[0]

    @property
    def C(self) -> int:
        return self.embedding.shape[1]

    def log_prob_unnorm(self, lprobs: Array) -> Array:  # (C,) -> ()
        M = (self.embedding * jnp.exp(lprobs)) @ self.embedding.T
        return jnp.linalg.slogdet(M)[1]

    def log_prob(self, lprobs: Array) -> Array:  # (C,) -> ()
        return self.log_prob_unnorm(lprobs) - self.log_prob_unnorm(
            jnp.zeros_like(lprobs)
        )

    def sample_one(
        self, active: Array, key: PRNGKey
    ) -> Tuple[
        Array, Array
    ]:  # Array[bool], PRNGKey -> index (0 <= i < C), prob (float 0 <= p <= 1)
        embed = self.embedding
        for i in range(len(active)):
            embed = jnp.where(active[i], reject(embed, embed[:, i]), embed)
        sqnorms = jnp.einsum("...ij,...ij->...j", embed, embed)
        probs = sqnorms / jnp.sum(sqnorms)
        choice = jax.random.choice(key, jnp.arange(len(active)), p=probs)
        return choice, probs[choice]

    @jax.jit
    def sample(
        self, key: PRNGKey
    ) -> Tuple[Array, Array]:  # PRNGKey -> Array[bool], float
        D = self.D
        C = self.C
        result = jnp.zeros((C,), dtype=jnp.bool)
        final_prob = jnp.array(1.0)
        for subkey in jax.random.split(key, D):
            choice, prob = self.sample_one(result, subkey)
            result = result.at[choice].set(True)
            final_prob *= prob
        return result, final_prob


@chex.dataclass
class Model:
    det_node: DetNode
    means: Array  # shape: (C, N)

    @property
    def N(self) -> int:
        return self.means.shape[1]

    def sample(self, key: PRNGKey) -> Tuple[Array, Array]:
        choice, prob = self.det_node.sample(key)
        return jnp.sum(self.means, axis=0, where=choice), prob
