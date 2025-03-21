import jax
import jax.numpy as jnp
import numpy as np

from jax.tree_util import tree_map
from functools import partial
from jax import random
from jax.random import PRNGKey
from einshape import jax_einshape as einshape
import operator

from beta_mnist import load_mnist
from lib import DetNode, BetaModel, BetaLeaf, GaussLeaf
import vae.utils as vae_utils

def dnode_probs(dnode):
    embedding = dnode.embedding
    probs = jnp.sum(embedding**2, axis=0)
    return probs


def train(model, mnist):
    pixel_count = mnist.image.shape[-1]
    @jax.value_and_grad
    @jax.jit
    def loss_and_grad(model, mnist, key):
        zero_image = jnp.zeros(pixel_count)
        lnprobs = jax.vmap(model.log_prob)(mnist.pos, mnist.neg)
        #partitions = jax.vmap(model.posterior_partition)(mnist.pos, mnist.neg, random.split(key, len(mnist.pos)))
        #entropies = jax.vmap(model.stochastic_entropy)(random.split(key, 1000))
        lnprobs = lnprobs #- partitions
        return -jnp.mean(lnprobs) #- jnp.mean(entropies)

    @jax.value_and_grad
    @jax.jit
    def entropy_loss_grad(model, key):
        entropies = jax.vmap(model.stochastic_entropy)(random.split(key, 100))
        return jnp.mean(entropies)

    @jax.grad
    @jax.jit
    def gradlognormalizer(model):
        zero_image = jnp.zeros(pixel_count)
        lognormalizer = model.log_prob(zero_image, zero_image)
        return lognormalizer

    model = model.replace(dnode=model.dnode.replace(
        embedding=model.dnode.embedding / jnp.linalg.norm(model.dnode.embedding, axis=0, keepdims=True)
    ))
    model = model.replace(dnode=model.dnode.renormalize())
    model = model.replace(leaves=model.leaves.clamp())
    BATCH_SIZE = 3000
    def get_batch(mnist, key):
        perm = random.permutation(key, len(mnist.image))
        indices = perm[:BATCH_SIZE]
        return tree_map(operator.itemgetter(indices), mnist)
    for i in range(50):
        key = PRNGKey(i)
        batch = get_batch(mnist, key)
        loss, grad = loss_and_grad(model, batch, key)
        #ent, egrad = entropy_loss_grad(model, key)
        #lngrad = gradlognormalizer(model)
        print(f'iter {i}, pixel loss {(loss)/ pixel_count:.4f}, {model.dnode.cluster_sizes()}')
        #model = tree_map(lambda p, g: p - 1e-1 * g, model, grad)
        leaves = model.leaves
        scale = grad.leaves.probe
        leaves = tree_map(lambda p, g, f: p - 1e-1 * (g) / (1e-32 + jnp.abs(scale*f)), leaves, grad.leaves, leaves.fisher)
        leaves = leaves.clamp()
        model = model.replace(leaves=leaves)
        
        #dnode = model.dnode
        #dnode = tree_map(lambda p, g, lng: p - 3e-2 * g, dnode, grad.dnode, lngrad.dnode)
        #dnode = dnode.renormalize()
        #model = model.replace(dnode=dnode)
        model = model.dnode_observe(batch.pos, batch.neg)

        prior = model.prior
        prior = tree_map(lambda p, g, f: p - 1e-1 * (g) / (1e-8 + jnp.abs(f)), prior, grad.prior, prior.fisher)
        prior = prior.clamp()
        model = model.replace(prior=prior)
    print(model.dnode.cluster_compositions())
    return model

def show(model, mnist):
    sample = model.sample(PRNGKey(0))
    sample = einshape("...(ss)->...ss", sample, s=SIZE)
    print(sample.shape)
    print(sample)
    rowsize = 8
    samples = jax.vmap(model.sample)(random.split(PRNGKey(0), rowsize**2))
    samples = einshape("(rr)(ss)->(rr)ss1", samples, s=SIZE, r=rowsize)
    print(samples.shape)
    vae_utils.save_image(samples, "results/beta/samples.png", nrow=rowsize)
    total_leaves = tree_map(jnp.add, model.leaves, model.prior)
    leaf_samples = jax.vmap(total_leaves.sample)(random.split(PRNGKey(0), rowsize))
    print(leaf_samples.shape)
    leaf_samples = einshape("rcz->crz", leaf_samples, r=rowsize)
    leaf_samples = einshape("cr(ss)->(cr)ss1", leaf_samples, s=SIZE, r=rowsize)
    print(leaf_samples.shape)
    vae_utils.save_image(leaf_samples, "results/beta/leaves_sample.png", nrow=rowsize)

    probs = model.dnode.marginal_probabilities()
    order = jnp.argsort(probs, descending=True)
    leaves_mean = model.leaves.mean[order] # C x N
    leaves_mean = einshape("c(ss)->css1", leaves_mean, s=SIZE)
    print(jnp.max(model.leaves.precision))
    print(jnp.min(model.leaves.precision))
    print(jnp.mean(model.leaves.precision))
    leaves_prec = jnp.log1p(model.leaves.precision[order]) / 2.
    leaves_prec = einshape("c(ss)->css1", leaves_prec, s=SIZE)
    together = jnp.concatenate([leaves_mean, leaves_prec], axis=0)
    vae_utils.save_image(together, "results/beta/leaves.png", nrow=len(together)//2)


CLASSES = jnp.arange(10)
SIZE = 24
def main():
    jnp.set_printoptions(precision=2, suppress=True)
    mnist = load_mnist(classes=CLASSES, height=SIZE, width=SIZE)
    print(f"max pixel: {jnp.max(mnist.image)}, min pixel: {jnp.min(mnist.image)}")
    print(tree_map(jnp.shape, mnist))

    D = 2
    C = 24
    N = SIZE * SIZE

    dnode = DetNode.create(D=D, C=C, key=PRNGKey(0))
    #leaftype = BetaLeaf
    leaftype = GaussLeaf
    model = BetaModel.create_with_dnode(dnode, N=N, key=PRNGKey(1), leaftype=leaftype)
    #model = BetaLeaf.init(key=PRNGKey(2), shape=(N,))
    #show(model, mnist)
    model = train(model, mnist)
    show(model, mnist)

if __name__ == '__main__':
    main()
