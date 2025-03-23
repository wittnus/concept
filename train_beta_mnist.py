import jax
import jax.numpy as jnp
import numpy as np

from jax.tree_util import tree_map
from functools import partial
from jax import random
from jax.random import PRNGKey
from einshape import jax_einshape as einshape
import operator

from beta_mnist import load_mnist, load_encoded_mnist, load_vae_model, vae_decode
from lib import DetNode, BetaModel, BetaLeaf, GaussLeaf
import vae.utils as vae_utils

@jax.value_and_grad
@jax.jit
def loss_and_grad(model, mnist, key):
    lnprobs = jax.vmap(model.log_prob_x)(mnist.x)
    if hasattr(mnist, 'precision'):
        precision = mnist.precision
    else:
        precision = 32.
    dist = model.prior.with_mean_and_precision(mnist.x, precision)
    kl_div = jax.vmap(model.kl_divergence)(dist)
    local_intractabilities = jax.vmap(model.intractability_at_x)(mnist.x)
    intractability = 1e-0 * local_intractabilities.mean()
    #global_intractability = model.intractability()
    #intractability = 1e-0 * global_intractability
    return kl_div.mean() #+ intractability
    return -jnp.mean(lnprobs) + intractability

def train(model, mnist):
    pixel_count = mnist.x.shape[-1]

    BATCH_SIZE = 3000
    def get_batch(mnist, key):
        perm = random.permutation(key, len(mnist.x))
        indices = perm[:BATCH_SIZE]
        return tree_map(operator.itemgetter(indices), mnist)
    for i in range(100):
        key = PRNGKey(i)
        batch = get_batch(mnist, key)
        loss, grad = loss_and_grad(model, batch, key)
        print(f'iter {i}, pixel loss {(loss)/ pixel_count:.4f}, {model.dnode.cluster_sizes()}')
        print(f"intractability: {model.intractability()}")
        model = model.replace(prior=model.prior.newton_step(grad.prior, lr=3e-1))
        if i > 10:
            model = model.replace(leaves=model.leaves.newton_step(grad.leaves, lr=1e-1))
        model = model.dnode_observe_x(batch.x, key)
    print(model.dnode.cluster_compositions())
    return model

def show(model, mnist):
    rowsize = 8
    samples = jax.vmap(model.sample)(random.split(PRNGKey(0), rowsize**2))
    samples = jax.vmap(as_image)(samples.mean)
    vae_utils.save_image(samples, "results/beta/samples.png", nrow=rowsize)

    leaf_sampless = []
    for i in range(len(model.leaves.mean)):
        leaf_samples = jax.vmap(partial(model.leaf_sample, index=i))(random.split(PRNGKey(0), rowsize))
        leaf_samples = leaf_samples.mean
        #leaf_samples = leaf_samples.sample(PRNGKey(1))
        leaf_samples = jax.vmap(as_image)(leaf_samples)
        leaf_sampless.append(leaf_samples)
    leaf_samples = jnp.concatenate(leaf_sampless, axis=0)
    #total_leaves = tree_map(jnp.add, model.leaves, model.prior)
    #leaf_samples = jax.vmap(total_leaves.sample, out_axes=1)(random.split(PRNGKey(0), rowsize))
    #leaf_samples = jax.vmap(as_image)(einshape("cr...->(cr)...", leaf_samples))
    print(leaf_samples.shape)
    vae_utils.save_image(leaf_samples, "results/beta/leaves_sample.png", nrow=rowsize)

    probs = model.dnode.marginal_probabilities()
    order = jnp.argsort(probs, descending=True)
    order = jnp.arange(len(probs))
    leaves_mean = model.leaves.mean[order] # C x N
    leaves_prec = model.leaves.precision[order] # C x N
    leaves_mean = jax.vmap(as_image)(leaves_mean)
    leaves_prec = jax.vmap(as_image)(jnp.log1p(leaves_prec) / 2.)
    together = jnp.concatenate([leaves_mean, leaves_prec], axis=0)
    vae_utils.save_image(together, "results/beta/leaves.png", nrow=len(together)//2)

def analyze_interference(model, mnist):
    M = model.interference_matrix()
    #print(np.array(M))
    eigvals, eigvecs = np.linalg.eigh(M)
    for i in range(-1, -4, -1):
        print(f"eigval {i}: {eigvals[i]}")
        print(f"eigvec {i}: {eigvecs[:,i]}")
        print(f"subset: {np.nonzero(eigvecs[:,i] > 0.)}")
    U = model.cointerference_matrix()
    #print(np.array(U))
    eigvals, eigvecs = np.linalg.eigh(U)
    for i in range(-1, -4, -1):
        print(f"eigval {i}: {eigvals[i]}")
        print(f"eigvec {i}: {eigvecs[:,i]}")
        print(f"subset: {np.nonzero(eigvecs[:,i] > 0.)}")


CLASSES = jnp.arange(5)
SIZE = 20
PIXEL_SPACE = True

if not PIXEL_SPACE:
    _MODEL, _PARAMS = load_vae_model()
def as_image(sample):
    if not PIXEL_SPACE:
        sample = vae_decode(_MODEL, _PARAMS, sample)
    return einshape("...(ss)->...ss1", sample, s=SIZE)

def main():
    jnp.set_printoptions(precision=2, suppress=True)
    if PIXEL_SPACE:
        mnist = load_mnist(classes=CLASSES, height=SIZE, width=SIZE)
        N = SIZE * SIZE
    else:
        mnist = load_encoded_mnist(classes=CLASSES)
        N = mnist.x.shape[-1]
    print(f"max pixel: {jnp.max(mnist.x)}, min pixel: {jnp.min(mnist.x)}")
    print(tree_map(jnp.shape, mnist))

    D = 2
    C = 20

    dnode = DetNode.create(D=D, C=C, key=PRNGKey(0))
    leaftype = BetaLeaf
    #leaftype = GaussLeaf
    model = BetaModel.create_with_dnode(dnode, N=N, key=PRNGKey(1), leaftype=leaftype)
    #model = model.replace(dnode=model.dnode.orthogonalize())

    cpm = model.cross_prob_matrix()
    print(cpm)
    print(model.interference_matrix())
    print(model.intractability())

    model = train(model, mnist)
    show(model, mnist)
    analyze_interference(model, mnist)
    exit()

if __name__ == '__main__':
    main()
