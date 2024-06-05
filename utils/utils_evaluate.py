import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from geomjax.rmhmc.integrators import implicit_midpoint
from geomjax.rmhmc.metrics import gaussian_riemannian

import ot
import os

path = os.path.dirname(__file__)


def wasserstein_distance(samples1, samples2, distance_fn):
    M = np.asarray(distance_fn(jnp.asarray(samples1), jnp.asarray(samples2)))
    return ot.emd2([], [], M, numItermax=1e10)


def evaluate(
    rng_key,
    samples,
    true_samples,
    repeats,
    subsample_size=2000,
):
    assert samples.shape == true_samples.shape
    num_samples = samples.shape[0]
    rng_keys = jr.split(rng_key, repeats)
    distances1 = []
    distances2 = []

    for rng_key in rng_keys:

        rng_key1, rng_key2 = jr.split(rng_key)
        indexes1 = jr.choice(rng_key1, jnp.arange(num_samples), (subsample_size,))
        indexes2 = jr.choice(rng_key2, jnp.arange(num_samples), (subsample_size,))

        distance_fn = lambda samples1, samples2: ot.dist(
            samples1, samples2, metric="euclidean"
        )

        distances1.append(
            wasserstein_distance(
                np.asarray(samples[indexes1]),
                np.asarray(true_samples[indexes2]),
                distance_fn,
            )
        )
        distances2.append(
            wasserstein_distance(
                np.asarray(true_samples[indexes1]),
                np.asarray(true_samples[indexes2]),
                distance_fn,
            )
        )

    distances1 = np.array(distances1)
    distances2 = np.array(distances2)
    print(
        f"Wasserstein distance to true samples: {[np.round(np.mean(distances1), 2), np.round(np.std(distances1), 2)]}"
    )
    print(
        f"Wasserstein distance between true samples: {[np.round(np.mean(distances2), 2), np.round(np.std(distances2), 2)]}"
    )

    return distances1, distances2


def number_gradient_evaluations(
    sampler_type,
    total_num_steps,
    num_chains,
    num_integration_steps,
    info,
    average_implicit_steps=5,
):
    if sampler_type in ["hmc", "lmc", "lmcmonge", "lmcmongeid"]:
        gradient_evals = total_num_steps * num_chains * num_integration_steps
    elif sampler_type in ["rmhmc"]:
        gradient_evals = (
            total_num_steps
            * num_chains
            * num_integration_steps
            * average_implicit_steps
        )
    elif (
        sampler_type in ["nuts", "nutslmc", "nutslmcmonge", "nutslmcmongeid"]
        or "chees" in sampler_type
    ):
        gradient_evals = jnp.sum(info.num_integration_steps)
    elif sampler_type in ["nutsrmhmc"]:
        gradient_evals = jnp.sum(info.num_integration_steps) * average_implicit_steps

    return gradient_evals


def estimate_implicit_steps(positions, momentums, step_size, logdensity_fn, metric_fn):

    integration_states = (positions, momentums)
    integrator = implicit_midpoint
    (
        _,
        kinetic_energy_fn,
        _,
        inverse_metric_vector_product,
    ) = gaussian_riemannian(metric_fn)
    symplectic_integrator = integrator(
        logdensity_fn,
        kinetic_energy_fn,
        inverse_metric_vector_product,
        return_info=True,
    )
    integrator_fn = lambda x: symplectic_integrator(x, step_size)
    _, info = jax.vmap(integrator_fn)(integration_states)
    return info.iters.mean()
