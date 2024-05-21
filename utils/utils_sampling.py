import jax
import time
import numpy as np


def inference_loop_multiple_chains(
    rng_key, sampler, initial_states, num_samples, num_chains
):
    # Assume all chains start at same possition
    kernel = sampler.step
    keys = jax.random.split(rng_key, num_chains)

    @jax.jit
    def one_step(states, rng_key):
        keys = jax.random.split(rng_key, num_chains)
        states, info = jax.vmap(kernel)(keys, states)
        return states, (states, info)

    start_time = time.time()
    keys = jax.random.split(rng_key, num_samples)
    _, (states, info) = jax.lax.scan(one_step, initial_states, keys)
    end_time = time.time()
    elapsed_time = end_time - start_time

    return states, info, elapsed_time


# Multiple chains with pmap
def inference_loop(rng_key, sampler, initial_state, num_samples):
    kernel = sampler.step

    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, info) = jax.lax.scan(one_step, initial_state, keys)

    return states, info


def inference_loop_multiple_chains_pmap(
    rng_key, sampler, initial_states, num_samples, num_chains
):
    start_time = time.time()
    keys = jax.random.split(rng_key, num_chains)
    _inference_loop_multiple_chains = jax.pmap(
        inference_loop, in_axes=(0, None, 0, None), static_broadcasted_argnums=(1, 3)
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    states, info = _inference_loop_multiple_chains(
        keys, sampler, initial_states, num_samples
    )
    return states, info, elapsed_time


def get_reference_draws(M, name_model, num_samples):
    if M.name in ["NealFunnel", "Squiggle", "Rosenbrock"]:
        rng_key_true = jax.random.PRNGKey(42)
        samples_true = np.array(M.generate_samples(rng_key_true, num_samples))
    elif M.name == "Banana":
        in_path = f"data/reference_draws_unconstrained/{M.name}/reference_samples.npy"
        samples_true = np.load(in_path)
        n, d = samples_true.shape
        assert num_samples <= n
        samples_true = samples_true[:num_samples]
    elif M.name == "LogReg":
        # TODO: For different datasets
        in_path = (
            f"data/reference_draws_unconstrained/{name_model}/reference_samples.npy"
        )
        samples_true = np.load(in_path)
        n, d = samples_true.shape
        assert num_samples <= n
        samples_true = samples_true[:num_samples]
    else:
        # Load from posteriordb
        in_path = f"data/reference_draws_unconstrained/{M.name}/reference_samples.csv"
        samples_true = np.genfromtxt(in_path, delimiter=",", skip_header=1)
        n, d = samples_true.shape
        assert num_samples <= n
        samples_true = samples_true[:num_samples, 0 : M.D]
    return samples_true
