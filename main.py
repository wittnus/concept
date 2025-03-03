from lib import DetNode, Model
import jax
from jax import numpy as jnp
from jax.nn import one_hot
import numpy as np
from chex import Array
from typing import Tuple
from jax.random import PRNGKey
from itertools import combinations
from functools import partial
from tqdm import tqdm, trange
from jax.tree_util import tree_map

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("TkAgg")



def all_possible_choices(D, C):  # D: ways of choosing D items from C
    return list(combinations(range(C), D))


def choice_as_array(choice: Tuple[int], C: int) -> Array:
    arr = jnp.zeros(C, dtype=jnp.bool)
    arr = arr.at[jnp.array(choice)].set(True)
    return arr


def choice_array_as_tuple(choice: Array) -> Tuple[int]:
    return tuple(i for i in range(len(choice)) if choice[i])


def choice_prob(dnode: DetNode, choice: Array) -> Array:
    return jnp.exp(dnode.log_prob(jnp.log(choice.astype(jnp.float32))))


def sample_model(model: Model, key: PRNGKey, count: int) -> Tuple[Array, Array]:
    samples = list()
    for subkey in jax.random.split(key, count):
        sample, prob = model.sample(subkey)
        samples.append(sample)
    return np.array(samples)

def check_cond_lprob(dnode: DetNode, key: PRNGKey) -> None:
    sample, cond_lprob = dnode.sample(key)
    overall_lprob = dnode.log_prob(jnp.log(sample.astype(jnp.float32)))
    print(f"cond_lprob: {cond_lprob}, overall_lprob: {overall_lprob}")
    assert cond_lprob <= overall_lprob

def compare_probs(dnode: DetNode, key: PRNGKey) -> None:
    all_choices = all_possible_choices(dnode.D, dnode.C)
    overall_probs = {
        choice: choice_prob(dnode, choice_as_array(choice, dnode.C)) for choice in all_choices
    }
    for choice, prob in overall_probs.items():
        print(f"{choice_as_array(choice, dnode.C)}: {prob}")
    print(f"sum of probs: {sum(overall_probs.values())}")

    counts = dict()
    num_samples = 1000
    for subkey in tqdm(jax.random.split(key, num_samples)):
        sample, prob = dnode.sample(subkey)
        choice = choice_array_as_tuple(sample)
        counts[choice] = counts.get(choice, 0) + 1
    for choice in all_choices:
        print(f"{choice_as_array(choice, dnode.C)}: {counts.get(choice, 0) / num_samples}")

def sample_model(model: Model, key: PRNGKey, count: int) -> Array:
    samples = list()
    for subkey in jax.random.split(key, count):
        sample, prob = model.sample(subkey)
        samples.append(sample)
    return np.array(samples)

def visualize_model_samples(model: Model, key: PRNGKey, show: bool=True) -> None:
    samples = np.array(sample_model(model, key, 1000))
    plt.scatter(samples[:, 0], samples[:, 1])
    if show:
        plt.show()

def compare_models_at_samples_from1(model1: Model, model2: Model, key: PRNGKey, count: int) -> None:
    samples = sample_model(model1, key, count)
    lprobs1 = model1.exact_log_density(samples)
    lprobs2 = model2.exact_log_density(samples)
    for s, p1, p2 in zip(samples, lprobs1, lprobs2):
        print(f"{s} {p1:.3f} {p2:.3f}")
    KL = np.mean(lprobs1 - lprobs2)
    print(f"KL: {KL}")

def estimate_KL(model1: Model, model2: Model, key: PRNGKey, count: int) -> float:
    samples = sample_model(model1, key, count)
    lprobs1 = model1.exact_log_density(samples)
    lprobs2 = model2.exact_log_density(samples)
    KL = np.mean(lprobs1 - lprobs2)
    return KL

def make_model(D, C, N, key):
    key1, key2 = jax.random.split(key)
    dnode = DetNode.create(D, C, key1)
    model = Model.create_with_dnode(dnode, N, key2)
    return model

@jax.jit
def model_nll(model, data):
    return -model.exact_log_density(data).mean()

@jax.jit
def model_entropy(model, data):
    return -model.exact_entropy(data).mean()

@jax.jit
def model_exact_stochastic_entropy(model, data, key):
    neg_entropy, counts = model.exact_stochastic_entropy(data, key)
    return -neg_entropy.mean(), counts.mean(axis=0)

@jax.jit
def model_stochastic_entropy(model, data, key):
    neg_entropy, counts = model.stochastic_entropy(data, key)
    return -neg_entropy.mean(), counts.mean(axis=0)
    return -model.stochastic_entropy(data, key).mean()

def fit_model(model, data):
    @jax.jit
    @partial(jax.value_and_grad, has_aux=True)
    def loss_grad(model, data, key):
        return model_stochastic_entropy(model, data, key)
        return model_entropy(model, data), 0
        return model_nll(model, data), 0
    P = 100
    total_loss = 0.
    #total_exact_loss = 0.
    #total_exact_entropy = 0.
    for i, subkey in enumerate(jax.random.split(PRNGKey(0), 3000)):
        (loss, counts), grad = loss_grad(model, data, subkey)
        #print(counts)
        #exact_loss = jax.jit(model_nll)(model, data)
        #exact_entropy = jax.jit(model_entropy)(model, data)
        total_loss += loss
        #total_exact_loss += exact_loss
        #total_exact_entropy += exact_entropy
        if i % P == P-1:
            exact_loss = model_nll(model, data)
            exact_entropy = model_entropy(model, data)
            print(f"loss: {total_loss/P:.3f} (ent: {exact_entropy:.3f} exact: {exact_loss:.3f})")
            #_, counts = model_stochastic_entropy(model, data, subkey)
            #print(counts)
            total_loss = 0.
            #total_exact_loss = 0.
            #total_exact_entropy = 0.
        model = tree_map(lambda p, g: p - 3e-2 * g, model, grad).renormalize_dnode()
    return model


def main():
    init_key = PRNGKey(1)
    D = 2
    C = 4
    N = 2
    dnode = DetNode.create(D, C, init_key)
    manual_embedding = jnp.array([one_hot(0,D), one_hot(0,D), one_hot(1,D), one_hot(1,D)]).T
    dnode = DetNode(embedding=manual_embedding)

    check_cond_lprob(dnode, PRNGKey(3))
    compare_probs(dnode, PRNGKey(2))

    model1 = Model.create_with_dnode(dnode, N, PRNGKey(4))
    visualize_model_samples(model1, PRNGKey(5))
    #model1 = make_model(D, C, N, PRNGKey(4))
    model2 = make_model(D+1, C+2, N, PRNGKey(6))
    compare_models_at_samples_from1(model1, model2, PRNGKey(8), 10)

    data_from_model1 = sample_model(model1, PRNGKey(7), 10)
    #model2 = fit_model(model1, data_from_model1)
    model2 = fit_model(model2, data_from_model1)
    nll1 = model_nll(model1, data_from_model1)
    nll2 = model_nll(model2, data_from_model1)
    print(f"nll1: {nll1}, nll2: {nll2}")
    compare_models_at_samples_from1(model1, model2, PRNGKey(8), 10)

    KL = estimate_KL(model1, model2, PRNGKey(9), 1000)
    print(f"Estimated KL: {KL}")
    visualize_model_samples(model1, PRNGKey(10), show=False)
    visualize_model_samples(model2, PRNGKey(10), show=True)



if __name__ == "__main__":
    main()
