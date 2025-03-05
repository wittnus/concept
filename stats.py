import jax
from jax.lax import lgamma
import jax.numpy as jnp

def gamma_surprise(base, increment):
    return lgamma(base + increment) - lgamma(base)

#def gamma_surprise(base, increment):
#    return _gamma_surprise(base, increment) - _gamma_surprise(jnp.ones_like(base), increment)

def dirichlet_surprise(base, increment):
    term1 = jnp.sum(gamma_surprise(base, increment))
    term2 = gamma_surprise(jnp.sum(base), jnp.sum(increment))
    return term1 - term2

def hyp_compare(base1, base2, increment):
    surprise1 = dirichlet_surprise(base1, increment)
    surprise2 = dirichlet_surprise(base2, increment)
    return surprise1 - surprise2

def hyp_same(base, increment1, increment2):
    surprise1 = dirichlet_surprise(base, increment1)
    surprise2 = dirichlet_surprise(base, increment2)
    surprise3 = dirichlet_surprise(2.*base, increment1 + increment2)
    return surprise3 - surprise1 - surprise2

def test_multinomial_same(increment1, increment2):
    """compare counts of samples from two multinomials, return log probability of being same"""
    D = len(increment1)
    base = jnp.ones(D)/D
    return hyp_same(base, increment1, increment2)

def test_multinomial_exact(probs, increment):
    """return probability of multinomial having given probs given increment"""
    D = len(probs)
    base = jnp.ones(D)/D
    logprob_general = dirichlet_surprise(base, increment)
    logprob_specific = jax.scipy.special.xlogy(increment, probs).sum()
    return logprob_specific - logprob_general

def main():
    base = jnp.array([1., 1.])
    base2 = 100.*jnp.array([1., 3., 3.])
    base2 = 1e4*jnp.array([5., 5.])
    #increment = 10.*jnp.array([1., 2., 3.])
    increment = 10.*jnp.array([7., 3.])
    sup1 = dirichlet_surprise(base, increment)
    sup2 = dirichlet_surprise(base2, increment)
    diff = hyp_compare(base, base2, increment)
    prob = jax.nn.sigmoid(diff)
    print(f"Base 1: {base} Base 2: {base2} Increment: {increment}")
    print(f"Surprise 1: {sup1:.3f} Surprise 2: {sup2:.3f} Difference: {diff:.3f}")
    print(f"Probability of 1: {prob:.4f}")

    D = 2
    base = jnp.ones(D)/D
    increment1 = 10.*(jnp.arange(D) + 1)
    increment2 = increment1
    #increment2 = increment1[::-1]
    diff = hyp_same(base, increment1, increment2)
    prob = jax.nn.sigmoid(diff)
    print(f"Base: {base} Increment 1: {increment1} Increment 2: {increment2}")
    print(f"Difference: {diff:.3f} Probability are same: {prob:.4f}")

    probs = increment1/increment1.sum()
    prob = jax.nn.sigmoid(test_multinomial_exact(probs, increment2))
    print(f"Probabilities: {probs} Increment 2: {increment2}")
    print(f"Probability of same: {prob:.4f}")



if __name__ == '__main__':
    main()
