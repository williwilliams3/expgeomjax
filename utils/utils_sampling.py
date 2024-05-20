import jax
import time
import numpy as np


def inference_loop(rng_key, sampler, initial_position, num_samples):
    initial_state = sampler.init(initial_position)

    kernel = sampler.step

    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


def inference_loop_multiple_chains(
    rng_key, sampler, initial_states, num_samples, num_chains
):
    # Assume all chains start at same possition
    kernel = sampler.step
    keys = jax.random.split(rng_key, num_chains)

    @jax.jit
    def one_step(states, rng_key):
        keys = jax.random.split(rng_key, num_chains)
        states, _ = jax.vmap(kernel)(keys, states)
        return states, states

    start_time = time.time()
    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_states, keys)
    end_time = time.time()
    elapsed_time = end_time - start_time

    return states, elapsed_time


def get_reference_draws(M, name_model):
    if M.name in ["NealFunnel", "Squiggle", "Rosenbrock"]:
        rng_key_true = jax.random.PRNGKey(42)
        samples_true = np.array(M.generate_samples(rng_key_true))
    elif M.name == "Banana":
        in_path = f"data/reference_draws_unconstrained/{M.name}/reference_samples.npy"
        samples_true = np.load(in_path)
    elif M.name == "LogReg":
        # TODO: For different datasets
        in_path = (
            f"data/reference_draws_unconstrained/{name_model}/reference_samples.npy"
        )
        samples_true = np.load(in_path)
    else:
        # Load from posteriordb
        in_path = f"data/reference_draws_unconstrained/{M.name}/reference_samples.csv"
        samples_true = np.genfromtxt(in_path, delimiter=",", skip_header=1)
        samples_true = samples_true[:, 0 : M.D]
    return samples_true
