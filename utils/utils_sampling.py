import jax
import time
import numpy as np
import os

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(os.path.dirname(current_file_path))


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
        state, info = kernel(rng_key, state)
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
    states, info = _inference_loop_multiple_chains(
        keys, sampler, initial_states, num_samples
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    return states, info, elapsed_time


def get_reference_draws(M, name_model, num_samples, sub_name=""):
    if name_model in ["funnel", "squiggle", "rosenrbrock"]:
        rng_key_true = jax.random.PRNGKey(42)
        samples_true = np.array(M.generate_samples(rng_key_true, num_samples))
    elif name_model == "banana":
        in_path = f"data/reference_samples/{name_model}/reference_samples.npy"
        in_path = os.path.join(current_directory, in_path)
        samples_true = np.load(in_path)
        n, d = samples_true.shape
        assert num_samples <= n
        samples_true = samples_true[:num_samples]
    elif name_model == "logreg":
        in_path = (
            f"data/reference_samples/{name_model}/{sub_name}/reference_samples.npy"
        )
        in_path = os.path.join(current_directory, in_path)
        samples_true = np.load(in_path)
        n, d = samples_true.shape
        assert num_samples <= n
        samples_true = samples_true[:num_samples]
    elif name_model == "postdb":
        # Load from posteriordb
        in_path = (
            f"data/reference_samples/{name_model}/{sub_name}/reference_samples.csv"
        )
        in_path = os.path.join(current_directory, in_path)
        samples_true = np.genfromtxt(in_path, delimiter=",", skip_header=1)
        n, d = samples_true.shape
        assert num_samples <= n
        samples_true = samples_true[:num_samples, 0 : M.D]
    return samples_true
