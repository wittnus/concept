import numpy as np
import jax
from jax import numpy as jnp
import chex
from chex import Array
from jax.random import PRNGKey
from typing import Tuple
from itertools import combinations


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

    @classmethod
    def create(cls, D: int, C: int, key: PRNGKey) -> "DetNode":
        embedding = jax.random.normal(key, (D, C))
        return cls(embedding=embedding).renormalize()

    def renormalize(self) -> "DetNode":
        M = self.embedding @ self.embedding.T
        invroot = jnp.linalg.inv(jnp.linalg.cholesky(M))
        embedding = invroot @ self.embedding
        return DetNode(embedding=embedding)

    @property
    def D(self) -> int:
        return self.embedding.shape[0]

    @property
    def C(self) -> int:
        return self.embedding.shape[1]

    def log_prob_unnorm(self, lprobs: Array) -> Array:  # (C,) -> ()
        M = (self.embedding * jnp.exp(lprobs)[...,None,:]) @ self.embedding.T
        M = M + 1e-3*jnp.eye(self.D)
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


def all_possible_choice_arrays(dnode: DetNode) -> Array:
    D = dnode.D
    C = dnode.C
    tuples = list(combinations(range(C), D))
    result = jnp.zeros((len(tuples), C), dtype=jnp.bool)
    for i, t in enumerate(tuples):
        result = result.at[i, t].set(True)
    return result


def unit_gaussian_log_density(x: Array) -> Array:
    """include quadratic term as well as normalization"""
    return -0.5 * jnp.sum(x ** 2, axis=-1) - 0.5 * x.shape[-1] * jnp.log(2 * jnp.pi)


@chex.dataclass
class Model:
    dnode: DetNode
    means: Array  # shape: (C, N)

    @classmethod
    def create_with_dnode(cls, dnode: DetNode, N: int, key: PRNGKey) -> "Model":
        means = jax.random.normal(key, (dnode.C, N)) * 1.
        return cls(dnode=dnode, means=means)

    def renormalize_dnode(self) -> "Model":
        return Model(dnode=self.dnode.renormalize(), means=self.means)

    def total_variance(self) -> Array:
        embedding = self.dnode.embedding
        means = self.means
        N = self.N
        probs = jnp.einsum("...ij,...ij->...j", embedding, embedding)
        outers = jnp.einsum("...ij,...ik->...ijk", means, means)
        expected_outer = jnp.einsum("...i,...ijk->...jk", probs, outers)
        return jnp.eye(N) + expected_outer

    def unnormalizer(self) -> Array:
        tvar_logdet = jnp.linalg.slogdet(self.total_variance())[1]
        factor = jnp.exp(0.5 * tvar_logdet / self.N)
        return jnp.eye(self.N) * factor
        return jnp.linalg.cholesky(self.total_variance())
        return jnp.real(jax.scipy.linalg.sqrtm(self.total_variance()))

    def normalizer(self) -> Array:
        return jnp.linalg.inv(self.unnormalizer())

    def decode_mean(self, choice: Array) -> Array:
        return jnp.sum(self.means, axis=-2, where=choice[..., None])

    @property
    def N(self) -> int:
        return self.means.shape[1]
    
    @jax.jit
    def sample(self, key: PRNGKey) -> Tuple[Array, Array]:
        key1, key2 = jax.random.split(key)
        choice, prob = self.dnode.sample(key1)
        center = self.decode_mean(choice)
        noise = jax.random.normal(key2, (self.N,))
        sample = center + noise
        sample = self.normalizer() @ sample
        return sample, prob

    def exact_log_density(self, x: Array) -> Array:
        #x = x @ jax.lax.stop_gradient(jax.lax.stop_gradient(self).unnormalizer().T)
        x = x @ self.unnormalizer().T
        all_choices = all_possible_choice_arrays(self.dnode)
        all_means = jax.vmap(self.decode_mean)(all_choices)
        choice_lprobs = self.dnode.log_prob(jnp.log(all_choices))
        diffs = x[...,None,:] - all_means
        log_densities = unit_gaussian_log_density(diffs)
        tvar_logdet = jnp.linalg.slogdet(self.total_variance())[1]
        return jax.scipy.special.logsumexp(log_densities + choice_lprobs, axis=-1) + 0.5 * tvar_logdet
