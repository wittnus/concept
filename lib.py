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

    def orthogonalize(self) -> "DetNode":
        c_idx = jnp.arange(self.C)[None, :]
        d_idx = jnp.arange(self.D)[:, None]
        mask = (c_idx * self.D) // self.C == d_idx
        return DetNode(embedding=jnp.where(mask, self.embedding, 0.0)).renormalize()

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

    def condition(self, lprobs: Array) -> "DetNode":
        weights = jnp.exp(0.5*lprobs)
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
        self, key: PRNGKey
    ) -> Tuple[Array, Array]:  # PRNGKey -> Array[bool], float
        D = self.D
        C = self.C
        result = jnp.zeros((C,), dtype=jnp.bool)
        cond_lprobs = jnp.zeros((C,))
        #final_prob = jnp.array(1.0)
        final_lprob = jnp.array(0.0)
        for subkey in jax.random.split(key, D):
            choice, lprob = self.sample_one_cond(result, cond_lprobs, subkey)
            result = result.at[choice].set(True)
            #final_prob *= prob
            final_lprob += lprob
        return result, final_lprob

    def observe(self, choices: Array) -> "DetNode":
        active_embeddings = jnp.where(choices[..., None, :], self.embedding, 0.0)
        Ms = jnp.einsum("...ij,...kj->...ik", active_embeddings, active_embeddings)
        @jax.vmap
        def invroot(M):
            return jnp.linalg.inv(jnp.real(jax.scipy.linalg.sqrtm(M + 1e-3*jnp.eye(self.D))))
        Minvroots = invroot(Ms)
        observed_embeddings = jnp.einsum("...ij,...jk->...ik", Minvroots, active_embeddings)
        total_embedding = jnp.sum(observed_embeddings, axis=-3)
        choice_counts = jnp.sum(choices, axis=-2)
        total_embedding = total_embedding / jnp.linalg.norm(total_embedding, axis=-2, keepdims=True)
        total_embedding = total_embedding * jnp.sqrt(choice_counts)
        new_node = DetNode(embedding=total_embedding).renormalize()
        new_embedding = self.embedding + 0.1*new_node.embedding

        new_embedding = new_embedding / jnp.linalg.norm(new_embedding, axis=-2, keepdims=True)
        new_embedding = new_embedding * jnp.sqrt(choice_counts)
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

