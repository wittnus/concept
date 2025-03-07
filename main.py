from lib import DetNode, Model
from stats import test_multinomial_same, test_multinomial_exact
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
import pandas as pd
import seaborn as sns
import vae.utils as vae_utils
import vae_lib as vae_lib
from collections import namedtuple

from PIL import Image

matplotlib.use("TkAgg")


def test_sampling_at_x(model: Model, key: PRNGKey, x: Array) -> None:
    x_unnorm = x @ model.unnormalizer().T
    all_choices = model.all_possible_choices()
    ground_truth_log_densities = model.all_choice_log_densities(x_unnorm)
    assert len(ground_truth_log_densities) == len(all_choices)
    ground_truth_probs = jax.nn.softmax(ground_truth_log_densities)
    #print(f"ground_truth_probs: {ground_truth_probs}")
    key1, key2 = jax.random.split(key)
    keys = jax.random.split(key1, 10000)
    exact_samples = jax.vmap(model.exact_cond_sample, in_axes=(None, 0))(x, keys)
    exact_counts = (all_choices == exact_samples[:, None, :]).all(axis=-1).sum(axis=0)
    exact_score = test_multinomial_exact(ground_truth_probs, exact_counts)
    #print(f"counts: {exact_counts}")
    gibbs_samples = exact_samples
    gibbs_samples = jax.vmap(model.greedy_sample_cond, in_axes=(None, 0))(x, keys)
    init_counts = (all_choices == gibbs_samples[:, None, :]).all(axis=-1).sum(axis=0)
    init_score = test_multinomial_exact(ground_truth_probs, init_counts)
    for i in range(30):
        key2, key3 = jax.random.split(key2)
        gibbs_keys = jax.random.split(key3, 10000)
        gibbs_samples = jax.vmap(model.gibbs_resample_cond, in_axes=(0, None, 0))(gibbs_samples, x_unnorm, gibbs_keys)
        #gibbs_samples = jax.vmap(model.adjacency_gibbs_cond, in_axes=(0, None, 0))(gibbs_samples, x_unnorm, gibbs_keys)

    gibbs_counts = (all_choices == gibbs_samples[:, None, :]).all(axis=-1).sum(axis=0)
    #print(f"gibbs_counts: {gibbs_counts}")
    gibbs_score = test_multinomial_exact(ground_truth_probs, gibbs_counts)
    mcmc_samples = model.monte_carlo_sample_cond(jnp.broadcast_to(x, (10000, *x.shape)), key2)
    mcmc_counts = (all_choices == mcmc_samples[:, None, :]).all(axis=-1).sum(axis=0)
    mcmc_score = test_multinomial_exact(ground_truth_probs, mcmc_counts)
    print(f"exact: {exact_score}, init: {init_score}, gibbs: {gibbs_score}, mcmc: {mcmc_score}")
    #print(f"done")



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
def model_entropy_at(model, data, choices):
    return -model.conditional_log_density(data, choices).mean()

@jax.jit
def model_exact_stochastic_entropy(model, data, key):
    neg_entropy, counts = model.exact_stochastic_entropy(data, key)
    return -neg_entropy.mean(), counts.mean(axis=0)

@jax.jit
def model_stochastic_entropy(model, data, key):
    neg_entropy, counts = model.monte_carlo_entropy(data, key)
    return -neg_entropy.mean(), counts.mean(axis=0)
    return -model.stochastic_entropy(data, key).mean()

def fit_model(model, data):
    @jax.jit
    @partial(jax.value_and_grad, has_aux=True)
    def loss_grad(model, data, key):
        return model_stochastic_entropy(model, data, key)
        return model_entropy(model, data), 0
        return model_nll(model, data), 0
    @jax.jit
    @jax.value_and_grad
    def cond_entropy(model, data, choices):
        return model_entropy_at(model, data, choices)
    P = 10
    total_loss = 0.
    #total_exact_loss = 0.
    #total_exact_entropy = 0.
    choices = model.monte_carlo_sample_cond(data, PRNGKey(0))
    opt_state = model.OptState.init(model)
    for i, subkey in enumerate(jax.random.split(PRNGKey(0), 100)):
        #choices = model.monte_carlo_resample_cond(data, choices, subkey, N=3)
        key1, key2 = jax.random.split(subkey)
        #choices = model.monte_carlo_sample_cond(data, key1)
        #choices = model.monte_carlo_resample_cond(data, choices, key2, N=100)
        choices = jax.vmap(model.exact_cond_sample)(data, jax.random.split(key1, data.shape[0]))
        loss, grad = cond_entropy(model, data, choices)
        #(loss, counts), grad = loss_grad(model, data, subkey)
        #print(counts)
        #exact_loss = jax.jit(model_nll)(model, data)
        #exact_entropy = jax.jit(model_entropy)(model, data)
        total_loss += loss
        #total_exact_loss += exact_loss
        #total_exact_entropy += exact_entropy
        #if i % 10 == 9:
        #    choices = model.monte_carlo_sample_cond(data, subkey)
        if i % P == P-1:
            exact_loss = model_nll(model, data)
            exact_entropy = model_entropy(model, data)
            print(f"loss: {total_loss/P:.3f} (ent: {exact_entropy:.3f} exact: {exact_loss:.3f})")
            #_, counts = model_stochastic_entropy(model, data, subkey)
            #print(counts)
            total_loss = 0.
            #total_exact_loss = 0.
            #total_exact_entropy = 0.
        #opt_state = model.observe(opt_state, data, choices)
        #model = model.apply(opt_state)
        model = model.cluster_var_observe(data, choices)
        #dnode = tree_map(lambda p, g: p - 1e-1 * g, model.dnode, grad.dnode).renormalize()
        dnode = model.dnode.observe(choices)
        model = model.replace(dnode=dnode)
        model = model.lstsq_observe(data, choices)
        #model = tree_map(lambda p, g: p - 3e-2 * g, model, grad).renormalize_dnode()
    return model

def compute_color_axis(mnist_data):
    colors = mnist_data["color"]
    encoding = mnist_data["encoding"]
    mean0 = encoding[colors == 0].mean(axis=0)
    mean1 = encoding[colors == 1].mean(axis=0)
    return mean1 - mean0

def compute_digit_axis(mnist_data, digit0, digit1=None):
    digits = mnist_data["digit"]
    encoding = mnist_data["encoding"]
    mean0 = encoding[digits == digit0].mean(axis=0)
    if digit1 is None:
        mean1 = encoding.mean(axis=0)
    else:
        mean1 = encoding[digits == digit1].mean(axis=0)
    return mean1 - mean0

def digit_color_project(mnist_data, encoding):
    color_axis = compute_color_axis(mnist_data)
    digit_axes = [compute_digit_axis(mnist_data, i) for i in range(5)]
    invprojection = jnp.array(digit_axes+[color_axis]).T
    projection = jnp.linalg.pinv(invprojection)
    return encoding @ projection.T

def make_dataframe(mnist_data):
    MAX = 1000
    digits = mnist_data["digit"][:MAX]
    colors = mnist_data["color"][:MAX]
    projected = digit_color_project(mnist_data, mnist_data["encoding"][:MAX])
    df = pd.DataFrame({
        "digit": [str(d) for d in digits],
        "color": ["black" if c == 0 else "white" for c in colors]
    })
    for i in range(projected.shape[1]):
        df[f"dim{i}"] = projected[:, i]
    return df[df["digit"].isin(["0", "1", "2", "3", "4"])]

def plot_projected(mnist_data, list_of_sets, digit=0):
    color_axis = compute_color_axis(mnist_data)
    #digit_axes = [compute_digit_axis(mnist_data, i) for i in range(1)]
    digit_axes = [compute_digit_axis(mnist_data, digit)]
    invprojection = jnp.array(digit_axes+[color_axis]).T
    projection = jnp.linalg.pinv(invprojection)
    for data in list_of_sets:
        projected = data @ projection.T
        plt.scatter(projected[:, 0], projected[:, -1])
    plt.show()

def pair_plot(mnist_data):
    df = make_dataframe(mnist_data)
    print(df.head())
    sns.pairplot(df, hue="digit", palette="husl", plot_kws=dict(size=df["color"]))
    plt.show()

def choose_for_concept(choices, concept_index, count=32, negate=False):
    key = PRNGKey(0)
    if negate:
        _choices = 1 - choices
    else:
        _choices = choices
    indices = jax.random.categorical(key, jnp.log(_choices[:, concept_index]), shape=(count,))
    return indices

def embed_covariance(embed):
    cov = embed.T @ embed
    cov = cov**2 / jnp.diag(cov)[:, None]
    return cov

def resample_concept(model, choices, concept_index, key):
    C = choices.shape[-1]
    cond_lprobs = jnp.log(1. - jax.nn.one_hot(concept_index, C))
    def resample(choice, key):
        choice = choice.at[concept_index].set(0)
        sample, _ = model.dnode.sample_one_cond(choice, cond_lprobs, key)
        choice = choice.at[sample].set(1)
        return choice
    new_choices = jax.vmap(resample)(choices, jax.random.split(key, len(choices)))
    #print(choices[:, concept_index].all(), ~new_choices[:, concept_index].any())
    #print((choices.sum(axis=-1) == 3).all(), (new_choices.sum(axis=-1) == 3).all())
    return new_choices

def apply_choice_diff(model, old_choices, new_choices, data):
    latent_diff = jax.vmap(model.decode_mean)(new_choices) - jax.vmap(model.decode_mean)(old_choices)
    latent_diff = latent_diff @ model.normalizer().T
    return data + latent_diff

class LatentDecoder:
    def __init__(self):
        config = namedtuple("Config", ["batch_size", "latents"])(batch_size=128, latents=20)
        model, params = vae_lib.load_model(config)
        self.model = model
        self.params = params
    def __call__(self, z):
        return vae_lib.decode(self.model, self.params, z)



def save_concepts(model, choices, z, recons, concept_embed, truncate=32, row_size=32):
    C = choices.shape[1]
    positive_indices = [choose_for_concept(choices, i, count=row_size) for i in range(C)]
    negative_indices = [choose_for_concept(choices, i, count=row_size, negate=True) for i in range(C)]
    concept_probs = choices.mean(axis=0)
    concept_ent = -jnp.log(concept_probs) * concept_probs
    sorted_indices = jnp.argsort(concept_ent, descending=True)[:truncate]
    np.set_printoptions(precision=2, suppress=True)
    sorted_choices = choices[:, sorted_indices]
    sorted_probs = concept_probs[sorted_indices]
    print(f"probs: {sorted_probs}")
    print(f"entropy: {-jnp.log(sorted_probs)*sorted_probs}")
    embed_cov = embed_covariance(concept_embed[:,sorted_indices])
    print(f"embed_cov:\n{embed_cov}")

    chosen_recons = jnp.concatenate([recons[positive_indices[i]] for i in sorted_indices])
    chosen_recons = chosen_recons.reshape(-1, 28, 28, 1)
    vae_utils.save_image(chosen_recons, "results/concepts.png", nrow=row_size)
    chosen_not_recons = jnp.concatenate([recons[negative_indices[i]] for i in sorted_indices])
    chosen_not_recons = chosen_not_recons.reshape(-1, 28, 28, 1)
    vae_utils.save_image(chosen_not_recons, "results/concepts_not.png", nrow=row_size)

    decoder = LatentDecoder()
    for i, concept_index in enumerate(sorted_indices):
        pchoices = choices[positive_indices[concept_index]]
        assert pchoices[:, concept_index].all()
        old_z = z[positive_indices[concept_index]]
        old_image = decoder(old_z)
        new_images = []
        for j in range(10):
            new_choices = resample_concept(model, pchoices, concept_index, PRNGKey(j))
            new_z = apply_choice_diff(model, pchoices, new_choices, old_z)
            new_image = decoder(new_z)
            new_images.append(new_image)
        all_images = jnp.concatenate([old_image] + new_images).reshape(-1, 28, 28, 1)
        vae_utils.save_image(all_images, f"results/concept_{i}.png", nrow=row_size)



def train_on_mnist():
    mnist_data = jnp.load("vae/results/encodings.npz")
    mnist_data = jnp.load("vae/results/encoded_ds.npz")
    #pair_plot(mnist_data)
    #exit()
    classes = mnist_data["digit"]
    colors = mnist_data["color"]
    allowed_classes = [0, 1, 2, 3]
    allowed_colors = [0, 1]
    only_ones = mnist_data["encoding"][classes == 1]
    only_zeros = mnist_data["encoding"][classes == 0]
    only_twos = mnist_data["encoding"][classes == 2]
    #plot_projected(mnist_data, [only_zeros, only_ones, only_twos])
    keep = (classes[:, None] == jnp.array(allowed_classes)).any(axis=-1)
    keep = keep & (colors[:, None] == jnp.array(allowed_colors)).any(axis=-1)
    data = mnist_data["encoding"][keep]
    recon = mnist_data["recon"][keep]
    print(f"data count: {len(data)}")
    COV = jnp.cov(data.T)
    print(f"covariance: {np.diag(COV)}")
    D = 2
    C = 10
    N = 20
    model = make_model(D, C, N, PRNGKey(4))
    nll0 = model_nll(model, data)
    print(f"nll0: {nll0}")
    model = fit_model(model, data)
    nll = model_nll(model, data)
    print(f"nll: {nll}")
    data_choice_samples = jax.vmap(model.exact_cond_sample)(data, jax.random.split(PRNGKey(0), len(data)))
    save_concepts(model, data_choice_samples, data, recon, model.dnode.embedding)
    exit()
    model_samples = sample_model(model, PRNGKey(5), len(data))
    for digit in allowed_classes:
        plot_projected(mnist_data, [data, model_samples], digit=digit)


def main():
    train_on_mnist()
    exit()
    init_key = PRNGKey(1)
    D = 1
    C = D*10
    N = 20
    dnode = DetNode.create(D, C, init_key)
    manual_embedding = jnp.array([one_hot(0,D), one_hot(0,D), one_hot(1,D), one_hot(1,D)]).T
    manual_embedding = jnp.array(
            sum([[one_hot(i, D)] * (C//D) for i in range(D)], [])
            ).T
    dnode = DetNode(embedding=manual_embedding)

    check_cond_lprob(dnode, PRNGKey(3))
    compare_probs(dnode, PRNGKey(2))

    model1 = Model.create_with_dnode(dnode, N, PRNGKey(4))
    model2 = make_model(D+0, C+4, N, PRNGKey(6))
    test_samples = sample_model(model1, PRNGKey(5), 3)
    print("testing in distribution sampling...")
    #for test_sample in test_samples:
    #    test_sampling_at_x(model1, PRNGKey(5), test_sample)
    print("testing out of distribution sampling...")
    #for test_sample in test_samples:
    #    test_sampling_at_x(model2, PRNGKey(5), test_sample)
    print("done testing sampling...")
    visualize_model_samples(model1, PRNGKey(5))
    #model1 = make_model(D, C, N, PRNGKey(4))
    #compare_models_at_samples_from1(model1, model2, PRNGKey(8), 10)

    data_from_model1 = sample_model(model1, PRNGKey(7), 1000)
    print(f"ground truth entropy: {-model1.exact_entropy(data_from_model1).mean()}")
    mnist_data = jnp.load("vae/results/encodings.npz")
    #data_from_model1 = mnist_data["encoding"].reshape(-1, 20)
    #model2 = fit_model(model1, data_from_model1)
    model2 = fit_model(model2, data_from_model1)
    nll1 = model_nll(model1, data_from_model1)
    nll2 = model_nll(model2, data_from_model1)
    print(f"nll1: {nll1}, nll2: {nll2}")
    #compare_models_at_samples_from1(model1, model2, PRNGKey(8), 10)

    KL = estimate_KL(model1, model2, PRNGKey(9), 1000)
    print(f"Estimated KL: {KL}")
    visualize_model_samples(model1, PRNGKey(10), show=False)
    visualize_model_samples(model2, PRNGKey(10), show=True)



if __name__ == "__main__":
    main()
