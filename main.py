from lib import DetNode, Model
import jax
from jax import numpy as jnp
from jax.nn import one_hot
import numpy as np
from chex import Array
from typing import Tuple
from jax.random import PRNGKey
from itertools import combinations
from tqdm import tqdm, trange

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


def main():
    init_key = PRNGKey(0)
    D = 2
    C = 4
    N = 2
    dnode = DetNode.create(D, C, init_key)
    #manual_embedding = jnp.array([one_hot(0,D), one_hot(0,D), one_hot(1,D), one_hot(1,D)]).T
    #dnode = DetNode(embedding=manual_embedding)

    sample_key = PRNGKey(3)
    sample, prob = dnode.sample(sample_key)
    print(sample, prob)
    cond_lprob = jnp.log(prob)
    overall_lprob = dnode.log_prob(jnp.log(sample.astype(jnp.float32)))
    print(f"cond_lprob: {cond_lprob}, overall_lprob: {overall_lprob}")
    assert cond_lprob <= overall_lprob

    all_choices = all_possible_choices(D, C)
    overall_probs = {
        choice: choice_prob(dnode, choice_as_array(choice, C)) for choice in all_choices
    }
    for choice, prob in overall_probs.items():
        print(f"{choice_as_array(choice, C)}: {prob}")
    print(f"sum of probs: {sum(overall_probs.values())}")

    counts = dict()
    num_samples = 1000
    for subkey in tqdm(jax.random.split(sample_key, num_samples)):
        sample, prob = dnode.sample(subkey)
        choice = choice_array_as_tuple(sample)
        counts[choice] = counts.get(choice, 0) + 1
    for choice in all_choices:
        print(f"{choice_as_array(choice, C)}: {counts.get(choice, 0) / num_samples}")

    model_init_key = PRNGKey(4)
    model = Model.create_with_dnode(dnode, N, model_init_key)
    model_sample_key = PRNGKey(5)
    samples = list()
    for subkey in tqdm(jax.random.split(model_sample_key, 1000)):
        sample, prob = model.sample(subkey)
        samples.append(sample)
    samples = np.array(samples)
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.show()


if __name__ == "__main__":
    main()
