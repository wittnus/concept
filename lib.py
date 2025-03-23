import numpy as np
import jax
from jax import numpy as jnp
import chex
from chex import Array
from jax.random import PRNGKey
from jax import random
from typing import Tuple
from itertools import combinations

from jax.scipy.special import xlogy, xlog1py, gammaln, logsumexp, betaln, polygamma, digamma
from jax.tree_util import tree_map
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

    @classmethod
    def create(cls, D: int, C: int, key: PRNGKey) -> "DetNode":
        embedding = jax.random.normal(key, (D, C))
        return cls(embedding=embedding).renormalize()

    def make_uniform(self) -> "DetNode":
        embedding = self.embedding / jnp.linalg.norm(self.embedding, axis=-2, keepdims=True)
        return self.replace(embedding=embedding).renormalize()

    def renormalize(self) -> "DetNode":
        M = self.embedding @ self.embedding.T
        invroot = jnp.linalg.inv(jnp.linalg.cholesky(M))
        embedding = invroot @ self.embedding
        return DetNode(embedding=embedding)

    def orthogonalize(self) -> "DetNode":
        c_idx = jnp.arange(self.C)[None, :]
        d_idx = jnp.arange(self.D)[:, None]
        mask = (c_idx * self.D) // self.C == d_idx
        return DetNode(embedding=jnp.where(mask, self.embedding, 0.0)).renormalize()

    def orthogonality_deficit(self) -> Array:
        Cov = self.embedding.T @ self.embedding
        Cov = jnp.abs(Cov)
        Cov = Cov @ Cov
        return jnp.sum(jnp.square(Cov)) - self.D

    def soft_orthogonalize(self) -> "DetNode":
        Cov = self.embedding.T @ self.embedding
        Cov = jnp.abs(Cov)
        #Cov = Cov @ Cov
        Cov = Cov / jnp.diag(Cov)[..., None]
        return DetNode(embedding=self.embedding @ Cov).renormalize()

    def marginal_probabilities(self) -> Array:
        return jnp.einsum("...ij,...ij->...j", self.embedding, self.embedding)

    def cluster_sizes(self) -> Array:
        Cov = self.embedding.T @ self.embedding
        Cov = jnp.abs(Cov)
        root_diag = jnp.sqrt(jnp.diag(Cov))
        Cov = Cov / root_diag[:, None] / root_diag[None, :]
        eigvals = jnp.linalg.eigvalsh(Cov)
        return eigvals[-self.D:]

    def cluster_compositions(self) -> Array:
        Cov = self.embedding.T @ self.embedding
        Cov = jnp.abs(Cov)
        root_diag = jnp.sqrt(jnp.diag(Cov))
        Cov = Cov / root_diag[:, None] / root_diag[None, :]
        eigvals, eigvecs = jnp.linalg.eigh(Cov)
        return jnp.square(eigvecs[:, -self.D:]).T

    @property
    def D(self) -> int:
        return self.embedding.shape[0]

    @property
    def C(self) -> int:
        return self.embedding.shape[1]

    def log_prob_unnorm(self, lprobs: Array) -> Array:  # (C,) -> ()
        M = (self.embedding * jnp.exp(lprobs)[...,None,:]) @ self.embedding.T
        M = M + 1e-4*jnp.eye(self.D)
        return jnp.linalg.slogdet(M)[1]

    def log_prob(self, lprobs: Array) -> Array:  # (C,) -> ()
        return self.log_prob_unnorm(lprobs) - self.log_prob_unnorm(
            jnp.zeros_like(lprobs)
        )

    def condition(self, lprobs: Array, eps=0.) -> "DetNode":
        weights = jnp.exp(0.5*lprobs)
        weights = jnp.maximum(weights, eps)
        embedding = self.embedding * weights[..., None, :]
        return DetNode(embedding=embedding).renormalize()

    def sample_one(
        self, active: Array, key: PRNGKey
    ) -> Tuple[
        Array, Array
    ]:  # Array[bool], PRNGKey -> index (0 <= i < C), prob (float 0 <= p <= 1)
        embed = self.embedding
        for i in range(len(active)):
            embed = jnp.where(active[i], reject(embed, embed[:, i]), embed)
        sqnorms = jnp.einsum("...ij,...ij->...j", embed, embed)
        probs = sqnorms / jnp.sum(sqnorms) * (~active).astype(jnp.float32)
        choice = jax.random.choice(key, jnp.arange(len(active)), p=probs)
        return choice, probs[choice]

    def sample_one_cond(self, active: Array, cond_lprobs: Array, key: PRNGKey) -> Tuple[Array, Array]:
        embed = self.embedding
        for i in range(len(active)):
            embed = jnp.where(active[i], reject(embed, embed[:, i]), embed)
        sqnorms = jnp.einsum("...ij,...ij->...j", embed, embed)
        lprobs = cond_lprobs + jnp.log(sqnorms)
        #lprobs = jnp.where(active, -jnp.inf, lprobs)
        choice = jax.random.categorical(key, lprobs, axis=-1)
        chosen_lprob = lprobs[choice] - jax.scipy.special.logsumexp(lprobs)
        return choice, chosen_lprob

    @jax.jit
    def sample(
        self, key: PRNGKey, cond_lprobs: Array = None
    ) -> Tuple[Array, Array]:  # PRNGKey -> Array[bool], float
        D = self.D
        C = self.C
        result = jnp.zeros((C,), dtype=jnp.bool)
        if cond_lprobs is None:
            cond_lprobs = jnp.zeros((C,))
        #final_prob = jnp.array(1.0)
        final_lprob = jnp.array(0.0)
        for subkey in jax.random.split(key, D):
            choice, lprob = self.sample_one_cond(result, cond_lprobs, subkey)
            result = result.at[choice].set(True)
            cond_lprobs = cond_lprobs.at[choice].set(-jnp.inf)
            #final_prob *= prob
            final_lprob += lprob
        return result, final_lprob

    def observe(self, choices: Array, float_obs: bool = False) -> "DetNode":
        if float_obs:
            active_embeddings = jnp.sqrt(choices[..., None, :]) * self.embedding
        else:
            active_embeddings = jnp.where(choices[..., None, :], self.embedding, 0.0)
        Ms = jnp.einsum("...ij,...kj->...ik", active_embeddings, active_embeddings)
        @jax.vmap
        def invroot(M):
            return jnp.linalg.inv(jnp.real(jax.scipy.linalg.sqrtm(M + 1e-3*jnp.eye(self.D))))
        Minvroots = invroot(Ms)
        observed_embeddings = jnp.einsum("...ij,...jk->...ik", Minvroots, active_embeddings)
        total_embedding = jnp.sum(observed_embeddings, axis=-3)
        choice_counts = jnp.sum(choices, axis=-2) + 1.
        total_embedding = total_embedding / (1. + jnp.linalg.norm(total_embedding, axis=-2, keepdims=True))
        total_embedding = total_embedding * jnp.sqrt(choice_counts)
        total_embedding = 0.9*total_embedding + 0.1*self.embedding
        new_node = DetNode(embedding=total_embedding).renormalize()
        new_embedding = self.embedding + 0.1*new_node.embedding

        #new_embedding = new_embedding / jnp.linalg.norm(new_embedding, axis=-2, keepdims=True)
        #new_embedding = new_embedding * jnp.sqrt(choice_counts)
        return DetNode(embedding=new_embedding).renormalize()




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

def all_choice_additions(choice: Array) -> Array:
    return jnp.eye(choice.shape[-1], dtype=jnp.bool) | choice[..., None, :]

def all_adjacencies(choice: Array) -> Array:
    to_add = jnp.eye(choice.shape[-1], dtype=jnp.bool)
    to_remove = jnp.eye(choice.shape[-1], dtype=jnp.bool)
    with_removed = ~to_remove[..., :, None, :] & choice[..., None, None, :]
    with_added = to_add[..., None, :, :] | with_removed
    return with_added
    


@chex.dataclass
class Model:
    dnode: DetNode
    means: Array  # shape: (C, N)
    invbasis: Array # shape: (N, N)

    @classmethod
    def create_with_dnode(cls, dnode: DetNode, N: int, key: PRNGKey) -> "Model":
        means = jax.random.normal(key, (dnode.C, N)) * 1.0
        invbasis = jnp.eye(N)
        return cls(dnode=dnode, means=means, invbasis=invbasis)

    def renormalize_dnode(self) -> "Model":
        return self.replace(dnode=self.dnode.renormalize())

    def total_variance(self) -> Array:
        embedding = self.dnode.embedding
        means = self.means
        N = self.N
        probs = jnp.einsum("...ij,...ij->...j", embedding, embedding)
        outers = jnp.einsum("...ij,...ik->...ijk", means, means)
        expected_outer = jnp.einsum("...i,...ijk->...jk", probs, outers)
        return jnp.eye(N) + expected_outer

    def total_mean(self) -> Array:
        embedding = self.dnode.embedding
        means = self.means
        probs = jnp.einsum("...ij,...ij->...j", embedding, embedding)
        return jnp.einsum("...i,...ij->...j", probs, means)

    def unnormalizer(self) -> Array:
        return self.invbasis
        tvar_logdet = jnp.linalg.slogdet(self.total_variance())[1]
        factor = jnp.exp(0.5 * tvar_logdet / self.N)
        return jnp.eye(self.N) #* factor
        return jnp.linalg.cholesky(self.total_variance())
        return jnp.real(jax.scipy.linalg.sqrtm(self.total_variance()))

    def normalizer(self) -> Array:
        return jnp.linalg.inv(self.unnormalizer())

    def decode_mean(self, choice: Array) -> Array:
        return jnp.sum(self.means, axis=-2, where=choice[..., None])

    def all_possible_choices(self) -> Array:
        return all_possible_choice_arrays(self.dnode)

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

    @jax.jit
    def sample_closest_gaussian(self, key:PRNGKey) -> Array:
        V = self.total_variance()
        Vchol = jnp.linalg.cholesky(V)
        noise = jax.random.normal(key, (self.N,))
        norm_noise = Vchol @ noise
        return self.normalizer() @ norm_noise

    @jax.jit
    def safe_sample(self, key: PRNGKey) -> Tuple[Array, Array]:
        key1, key2 = jax.random.split(key)
        all_choices = all_possible_choice_arrays(self.dnode)
        choice_lprobs = jax.vmap(self.dnode.log_prob)(jnp.log(all_choices))
        choice = jax.random.categorical(key1, choice_lprobs, axis=-1)
        center = self.decode_mean(all_choices[choice])# - self.total_mean()
        noise = jax.random.normal(key2, (self.N,))
        sample = center + noise
        sample = self.normalizer() @ sample
        return sample, choice_lprobs[choice]

    def choice_log_density(self, choice: Array, x: Array) -> Array:
        mean = self.decode_mean(choice)
        choice_log_prob = self.dnode.log_prob(jnp.log(choice))
        diff = x - mean
        return unit_gaussian_log_density(diff) + choice_log_prob

    def all_choice_log_densities(self, x) -> Array:
        """for internal use - does not normalize"""
        all_choices = all_possible_choice_arrays(self.dnode)
        return jax.vmap(self.choice_log_density, in_axes=(0, None), out_axes=-1)(all_choices, x)

    def exact_all_choice_log_densities(self, x) -> Array:
        """for external use"""
        x = x @ self.unnormalizer().T
        return self.all_choice_log_densities(x)

    def exact_log_density(self, x: Array) -> Array:
        #x = x @ jax.lax.stop_gradient(jax.lax.stop_gradient(self).unnormalizer().T)
        x = x @ self.unnormalizer().T
        choice_log_densities = self.all_choice_log_densities(x)
        marginal_density = jax.scipy.special.logsumexp(choice_log_densities, axis=-1)
        #tvar_logdet = jnp.linalg.slogdet(self.total_variance())[1]
        unnorm_logdet = jnp.linalg.slogdet(self.unnormalizer())[1]
        return marginal_density + unnorm_logdet

    def conditional_log_density(self, x: Array, choice: Array) -> Array:
        x = x @ self.unnormalizer().T
        choice_log_density = jax.vmap(self.choice_log_density)(choice, x)
        unnorm_logdet = jnp.linalg.slogdet(self.unnormalizer())[1]
        return choice_log_density + unnorm_logdet

    def exact_entropy(self, x: Array) -> Array:
        x = x @ self.unnormalizer().T
        lprobs = self.all_choice_log_densities(x)
        probs = jax.lax.stop_gradient(jax.nn.softmax(lprobs, axis=-1))
        marginal_entropy = jnp.sum(probs * lprobs, axis=-1)
        unnorm_logdet = jnp.linalg.slogdet(self.unnormalizer())[1]
        return marginal_entropy + unnorm_logdet

    @jax.jit
    def exact_cond_sample(self, x: Array, key: PRNGKey) -> Array:
        x = x @ self.unnormalizer().T
        all_choices = all_possible_choice_arrays(self.dnode)
        lprobs = self.all_choice_log_densities(x)
        choices = jax.random.categorical(key, lprobs, axis=-1)
        return all_choices[choices]

    def exact_stochastic_entropy(self, x: Array, key: PRNGKey) -> Array:
        x = x @ self.unnormalizer().T
        all_choices = all_possible_choice_arrays(self.dnode)
        lprobs = self.all_choice_log_densities(x)
        choices = jax.random.categorical(key, lprobs, axis=-1)
        choice_log_densities = jnp.take_along_axis(lprobs, choices[..., None], axis=-1)[..., 0]
        choice_counts = all_choices[choices]
        #choice_counts = jnp.take_along_axis(all_choices, choices[..., None], axis=-2)[..., 0, :]
        unnorm_logdet = jnp.linalg.slogdet(self.unnormalizer())[1]
        return choice_log_densities + unnorm_logdet, choice_counts

    @jax.jit
    def monte_carlo_resample_cond(self, x: Array, choices: Array, key: PRNGKey, N: int=10) -> Array:
        x = x @ self.unnormalizer().T
        def step(i, state):
            key, choices = state
            key, new_key = jax.random.split(key)
            new_choices = jax.vmap(self.gibbs_resample_cond)(choices, x, jax.random.split(key, x.shape[0]))
            return new_key, new_choices
        key, choices = jax.lax.fori_loop(0, N, step, (key, choices))
        return choices

    @jax.jit
    def monte_carlo_sample_cond(self, x: Array, key: PRNGKey) -> Array:
        x_unnorm = x @ self.unnormalizer().T
        init_key, gibbs_key = jax.random.split(key)
        choices = jax.vmap(self.greedy_sample_cond)(x_unnorm, jax.random.split(init_key, x.shape[0]))
        choices = self.monte_carlo_resample_cond(x, choices, gibbs_key)
        return choices

    def monte_carlo_entropy(self, x: Array, key: PRNGKey) -> Array:
        choices = self.monte_carlo_sample_cond(x, key)
        x = x @ self.unnormalizer().T
        choice_log_densities = jax.vmap(self.choice_log_density)(choices, x)
        logdet_term = jnp.linalg.slogdet(self.unnormalizer())[1]
        return choice_log_densities + logdet_term, choices.astype(jnp.float32)

    def stochastic_entropy(self, x: Array, key: PRNGKey) -> Array:
        #init_choices = self.exact_cond_sample(x, key)
        x = x @ self.unnormalizer().T
        logdet_term = jnp.linalg.slogdet(self.unnormalizer())[1]
        init_key, gibbs_key = jax.random.split(key)
        choices = jax.vmap(self.greedy_sample_cond)(x, jax.random.split(init_key, x.shape[0]))
        #choices = init_choices
        #jax.debug.print("choices: {}", choices)
        choice_log_densities = jax.vmap(self.choice_log_density)(choices, x)
        mass = 1
        choice_counts = choices.astype(jnp.float32)
        for i in range(30):
            gibbs_key, subkey = jax.random.split(gibbs_key)
            choices = jax.vmap(self.gibbs_resample_cond)(choices, x, jax.random.split(subkey, x.shape[0]))
            #print(jnp.mean(choices, axis=0))
            new_log_densities = jax.vmap(self.choice_log_density)(choices, x)
            choice_log_densities = new_log_densities
            choice_counts = choices.astype(jnp.float32)
            mass = 1
        #jax.debug.print("{}", (jnp.sum(choices.astype(jnp.int32), axis=-1)==self.dnode.D).all())
        return choice_log_densities / mass + logdet_term, choice_counts / mass

    def greedy_sample_one_cond(self, choice: Array, x: Array, key: PRNGKey) -> Array:
        new_choices = all_choice_additions(choice)
        new_log_densities = jax.vmap(self.choice_log_density, in_axes=(-2, None), out_axes=-1)(new_choices, x)
        new_log_densities = jnp.where(choice, -jnp.inf, new_log_densities)
        new_index = jax.random.categorical(key, new_log_densities, axis=-1)
        new_choice = choice.at[new_index].set(True)
        return new_choice

    def greedy_sample_cond(self, x: Array, key: PRNGKey) -> Array:
        choice = jnp.zeros((self.dnode.C,), dtype=jnp.bool)
        for i in range(self.dnode.D):
            choice = self.greedy_sample_one_cond(choice, x, key)
        return choice

    def gibbs_resample_cond(self, choice: Array, x: Array, key: PRNGKey) -> Array:
        key1, key2 = jax.random.split(key)
        deactivate = jax.random.categorical(key1, jnp.log(choice.astype(jnp.float32)), axis=-1)
        choice = choice.at[deactivate].set(False)
        choice = self.greedy_sample_one_cond(choice, x, key2)
        return choice

    @jax.jit
    def lstsq_observe(self, x: Array, choices: Array) -> "Model":
        x = x @ self.unnormalizer().T
        f_choices = choices.astype(jnp.float32)
        CC = f_choices.T @ f_choices
        project = jnp.linalg.inv(CC + jnp.eye(CC.shape[0]))
        new_means = project @ (f_choices.T @ x)
        return self.replace(means=new_means)

    def cluster_var_observe(self, x: Array, choices: Array) -> "Model":
        x = x @ self.unnormalizer().T
        mean = jax.vmap(self.decode_mean)(choices)
        errors = x - mean
        error_cov = jnp.cov(errors.T)
        root_cov = jnp.linalg.cholesky(error_cov + 1e-3*jnp.eye(self.N))
        new_means = self.means @ root_cov.T
        iroot = jnp.linalg.inv(root_cov)
        new_invbasis = iroot @ self.invbasis
        return Model(dnode=self.dnode, means=new_means, invbasis=new_invbasis)


@chex.dataclass
class Leaf:
    probe: Array # for computing observation counts

    @classmethod
    def default(cls, shape=()) -> "Leaf":
        return cls(probe=jnp.zeros(shape))

    @classmethod
    def init(cls, key, shape=()) -> "Leaf":
        return cls.default(shape=shape)

    @classmethod
    def sufficient_statistics(cls, x: Array) -> Tuple[Array, Array]:
        raise NotImplementedError

    @property
    def expectation(self) -> Tuple[Array, Array]:
        """expectation of sufficient statistics"""
        raise NotImplementedError

    def clamp(self, minval: float = 0.0, maxval: float = 1.0) -> "Leaf":
        return self

    @property
    def mean(self) -> Array:
        raise NotImplementedError

    @property
    def fisher(self) -> "Leaf":
        """i.e. variance of sufficient statistics"""
        raise NotImplementedError

    def newton_step(self, grad: "Leaf", lr=1e-1) -> "Leaf":
        count = grad.probe
        def update(param, grad, fisher):
            return param - lr * grad / (1e-32 + jnp.abs(count * fisher))
        return tree_map(update, self, grad, self.fisher).clamp()

    def partition(self) -> Array:
        raise NotImplementedError

    def log_prob(self, *t, prior: "Leaf"=None) -> Array:
        """log density as function of natural statistic"""
        probe = self.probe - jax.lax.stop_gradient(self.probe)
        if prior is None:
            return self._log_prob(*t) - self.partition() + probe
        else:
            total = tree_map(jnp.add, self, prior)
            return self._log_prob(*t) - total.partition() + prior.partition() + probe

    def cross_negentropy(self, other: "Leaf", prior: "Leaf"=None) -> Array:
        return self.log_prob(*other.expectation, prior=prior)

    def sample(self, key: PRNGKey) -> Array:
        raise NotImplementedError



@chex.dataclass
class BetaLeaf(Leaf):
    alpha: Array
    beta: Array

    @classmethod
    def default(cls, shape=()) -> "BetaLeaf":
        return cls(probe=jnp.zeros(shape), alpha=jnp.zeros(shape), beta=jnp.zeros(shape))

    @classmethod
    def init(cls, key, shape=()):
        key1, key2 = jax.random.split(key)
        alpha = jax.random.exponential(key1, shape) / 1e1
        beta = jax.random.exponential(key2, shape) / 1e1
        return cls.default(shape=shape).replace(alpha=alpha, beta=beta)

    @classmethod
    def with_mean_and_precision(cls, mean: Array, precision: Array) -> "BetaLeaf":
        alpha = mean * precision
        beta = (1. - mean) * precision
        return cls.default(shape=mean.shape).replace(alpha=alpha, beta=beta)

    @classmethod
    def sufficient_statistics(cls, x: Array) -> Tuple[Array, Array]:
        x = jnp.clip(x, 1e-3, 1. - 1e-3)
        return jnp.log(x), jnp.log1p(-x)

    @property
    def mean(self) -> Array:
        """measure of central tendency for display purposes"""
        return (self.alpha + 1.) / (self.alpha + self.beta + 2.)

    @property
    def expectation(self) -> Tuple[Array, Array]:
        """expectation of sufficient statistics"""
        exp_lnx = digamma(self.alpha + 1.) - digamma(self.alpha + self.beta + 2.)
        exp_ln1mx = digamma(self.beta + 1.) - digamma(self.alpha + self.beta + 2.)
        return exp_lnx, exp_ln1mx
        #return BetaLeaf(alpha=exp_lnx, beta=exp_ln1mx, probe=jnp.zeros_like(self.probe))

    @property
    def precision(self) -> Array:
        """measure of spread for display purposes"""
        return self.alpha + self.beta

    @property
    def fisher(self) -> "BetaLeaf":
        trig_alpha = polygamma(2, self.alpha + 1.)
        trig_beta = polygamma(2, self.beta + 1.)
        trig_sum = polygamma(2, self.alpha + self.beta + 2.)
        curv_alpha = trig_sum - trig_alpha
        curv_beta = trig_sum - trig_beta
        curv_probe = jnp.ones_like(self.probe)
        return BetaLeaf(alpha=curv_alpha, beta=curv_beta, probe=curv_probe)


    def clamp(self, minval: float = 0.0, maxval: float = 3e1) -> "BetaLeaf":
        alpha = jnp.clip(self.alpha, minval, maxval)
        beta = jnp.clip(self.beta, minval, maxval)
        return BetaLeaf(alpha=alpha, beta=beta, probe=jnp.zeros_like(self.probe))

    def partition(self) -> Array:
        return betaln(self.alpha + 1., self.beta + 1.)

    def _semi_partition(self) -> Array:
        return gammaln(self.alpha + 1.) + gammaln(self.beta + 1.)

    def _log_prob(self, lnx, ln1mx) -> Array:
        return self.alpha * lnx + self.beta * ln1mx

    def expected_log_prob_unnorm(self, other: "BetaLeaf") -> Array:
        return (self.alpha)*digamma(other.alpha + 1) + (self.beta)*digamma(other.beta + 1) - (self.alpha + self.beta)*digamma(other.alpha + other.beta + 2)

    def sample(self, key: PRNGKey) -> Array:
        return jax.random.beta(key, self.alpha + 1., self.beta + 1.)


@chex.dataclass
class GaussLeaf(Leaf):
    location: Array
    precision: Array
    probe: Array

    @classmethod
    def unit(cls, shape=()) -> "GaussLeaf":
        return cls(location=jnp.zeros(shape), precision=jnp.ones(shape), probe=jnp.zeros(shape))

    @classmethod
    def default(cls, shape=()) -> "GaussLeaf":
        return cls.unit(shape=shape)

    @classmethod
    def init(cls, key, shape=()):
        key1, key2 = jax.random.split(key)
        location = jax.random.normal(key1, shape)
        #location = jax.random.uniform(key1, shape)
        precision = jax.random.exponential(key2, shape) / 1e0
        return cls(location=location, precision=precision, probe=jnp.zeros(shape))

    @classmethod
    def with_mean_and_precision(cls, mean: Array, precision: Array) -> "GaussLeaf":
        location = precision * mean
        precision = jnp.broadcast_to(precision / 2., mean.shape)
        return cls(location=location, precision=precision, probe=jnp.zeros_like(mean))

    @classmethod
    def sufficient_statistics(cls, x: Array) -> Tuple[Array, Array]:
        return x, x**2

    @property
    def mean(self) -> Array:
        return self.location / (2. * self.precision + 1e-3)

    @property
    def expectation(self) -> Tuple[Array, Array]:
        return self.mean, self.mean**2 + 1. / (2. * self.precision + 1e-3)

    @property
    def fisher(self) -> "GaussLeaf":
        f = 1 / (2. * self.precision + 1e-3)
        return GaussLeaf(location=f, precision=f**2, probe=jnp.ones_like(self.probe))

    def clamp(self, minval: float = -1e1, maxval: float = 1e4) -> "GaussLeaf":
        location = jnp.clip(self.location, minval, maxval)
        precision = jnp.clip(self.precision, 1e-1, maxval)
        return GaussLeaf(location=location, precision=precision, probe=jnp.zeros_like(self.probe))

    def partition(self) -> Array:
        return -0.5 * jnp.log(self.precision / jnp.pi) + 0.25 * self.location**2 / (self.precision + 1e-3)

    def _semi_partition(self) -> Array:
        return -0.5 * jnp.log(self.precision / jnp.pi)

    def _log_prob(self, x: Array, xsq: Array, prior: "GaussLeaf"=None) -> Array:
        return self.location * x - self.precision * xsq
        probe = self.probe - jax.lax.stop_gradient(self.probe)
        lnprob = -self.precision * pos**2 + self.location * pos + probe
        if prior is None:
            return lnprob - self.partition()
        else:
            total = tree_map(jnp.add, self, prior)
            return lnprob - total.partition() + prior.partition()

    def sample(self, key: PRNGKey) -> Array:
        white = jax.random.normal(key, shape=self.location.shape)
        return self.mean + white / jnp.sqrt(self.precision * 2. + 1e-3)




@chex.dataclass
class BetaModel:
    dnode: DetNode
    prior: Leaf
    leaves: Leaf

    @classmethod
    def create_with_dnode(cls, dnode: DetNode, N: int, key: PRNGKey, leaftype=BetaLeaf) -> "BetaModel":
        return cls(
            dnode=dnode.make_uniform(),
            prior=leaftype.default((N,)),
            leaves=leaftype.init(key, (dnode.C, N)).clamp()
        )

    def leaf_log_probs(self, *t) -> Array:
        extended_ts = list(map(lambda a: a[..., None, :], t))
        return self.leaves.log_prob(*extended_ts, prior=self.prior).sum(axis=-1)

    def log_prob(self, *t) -> Array:
        prior_log_prob = self.prior.log_prob(*t).sum(axis=-1)
        leaves_log_prob = self.leaf_log_probs(*t)
        #uniform = logsumexp(leaves_log_prob, axis=-1) - jnp.log(leaves_log_prob.shape[-1])
        max_leaf_log_prob = jnp.max(leaves_log_prob, axis=-1)
        root_log_prob = self.dnode.log_prob(leaves_log_prob - max_leaf_log_prob)
        root_log_prob = root_log_prob + max_leaf_log_prob + prior_log_prob
        return root_log_prob

    def log_prob_x(self, x: Array) -> Array:
        return self.log_prob(*self.leaves.sufficient_statistics(x))

    def cross_entropy(self, dist: "Leaf") -> Array:
        return -self.log_prob(*dist.expectation)

    def kl_divergence(self, dist: "Leaf") -> Array:
        return -self.log_prob(*dist.expectation) + jnp.sum(dist.log_prob(*dist.expectation))

    def sample(self, key) -> Array:
        key1, key2 = jax.random.split(key)
        choices, _ = self.dnode.sample(key1)
        total_chosen = tree_map(lambda l: jnp.sum(l, axis=-2, where=choices[..., None]), self.leaves)
        total_chosen = tree_map(jnp.add, total_chosen, self.prior)
        return total_chosen

    def leaf_sample(self, key, index) -> Array:
        key1, key2 = jax.random.split(key)
        leaf_lprobs = jnp.zeros((self.dnode.C,)).at[index].set(jnp.inf)
        choices, _ = self.dnode.sample(key1, cond_lprobs=leaf_lprobs)
        total_chosen = tree_map(lambda l: jnp.sum(l, axis=-2, where=choices[..., None]), self.leaves)
        total_chosen = tree_map(jnp.add, total_chosen, self.prior)
        return total_chosen

    @jax.jit
    def dnode_observe_x(self, x: Array, key: PRNGKey) -> "BetaModel":
        leaves_log_prob = self.leaf_log_probs(*self.leaves.sufficient_statistics(x))
        leaves_log_prob = leaves_log_prob - jnp.max(leaves_log_prob, axis=-1, keepdims=True)
        leaf_choices = jax.vmap(self.dnode.sample)(random.split(key, x.shape[0]), cond_lprobs=leaves_log_prob)[0]
        #leaf_probs = jax.vmap(jax.grad(self.dnode.log_prob_unnorm))(leaves_log_prob)
        #dnode = self.dnode.observe(leaf_probs, float_obs=True)
        dnode = self.dnode.observe(leaf_choices, float_obs=False)
        return self.replace(dnode=dnode)

    def cross_prob_matrix(self) -> Array:
        return self.cond_cross_prob_matrix(jnp.zeros((self.dnode.C,)))

    def cond_cross_prob_matrix(self, cond_probs: Array) -> Array:
        C = self.dnode.C
        lprob_fn = self.dnode.log_prob_unnorm
        #lprob_fn = logsumexp
        probs = jax.grad(lprob_fn)(cond_probs)
        cross_probs = jax.hessian(lprob_fn)(cond_probs)
        outer_probs = jnp.einsum("...i,...j->...ij", probs, probs)
        return outer_probs + cross_probs - jnp.diag(probs)

    def cross_prob_matrix_at_x(self, x: Array) -> Array:
        leaves_log_prob = self.leaf_log_probs(*self.leaves.sufficient_statistics(x))
        leaves_log_prob = leaves_log_prob - jnp.max(leaves_log_prob, axis=-1, keepdims=True)
        return self.cond_cross_prob_matrix(leaves_log_prob)

    def interference_matrix(self) -> Array:
        total_leaves = tree_map(jnp.add, self.leaves, self.prior)
        cross_totals = tree_map(lambda a, b, p: a[:,None,:] + b[None,:,:] + p, self.leaves, self.leaves, self.prior)
        individual = (self.prior._semi_partition() - total_leaves._semi_partition()).sum(axis=-1)
        cross = (self.prior._semi_partition() - cross_totals._semi_partition()).sum(axis=-1)
        diff = individual[None, :] + individual[:, None] - cross
        return diff

    def cointerference_matrix(self) -> Array:
        total_leaves = tree_map(jnp.add, self.leaves, self.prior)
        cross_totals = tree_map(lambda a, b, p: a[:,None,:] + b[None,:,:] + p, self.leaves, self.leaves, self.prior)
        individual = -(self.prior.partition() - self.prior._semi_partition() - total_leaves.partition() + total_leaves._semi_partition()).sum(axis=-1)
        cross = -(self.prior.partition() - self.prior._semi_partition() - cross_totals.partition() + cross_totals._semi_partition()).sum(axis=-1)
        diff = individual[None, :] + individual[:, None] - cross
        return diff

    def _intractability(self, cpm):
        cpm.at[jnp.diag_indices(cpm.shape[-1])].set(0.0)
        cpm = jax.lax.stop_gradient(cpm)
        return jnp.sum(self.interference_matrix() * cpm)

    def intractability(self):
        return self._intractability(self.cross_prob_matrix())

    def intractability_at_x(self, x: Array):
        return self._intractability(self.cross_prob_matrix_at_x(x))







