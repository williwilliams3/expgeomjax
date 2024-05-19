from jax import config

config.update("jax_enable_x64", True)
import os
import sys
import copy
import gc

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import numpy as np
import jax
import jax.random as jr
import jax.numpy as jnp
import hydra
from omegaconf import OmegaConf
from src.utils_models import set_model
from src.utils_sampling import inference_loop_multiple_chains, get_reference_draws
from src.utils_sampler import set_sampler, set_params_sampler
from src.utils_metric import set_metric_fn
from src.utils_evaluate import evaluate
from plotting.plotting_functions import plot_samples_marginal


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg):
    print(OmegaConf.to_yaml(cfg))
    sampler_config = cfg.sampler
    model_config = cfg.model

    repeats = cfg.repeats
    seed = cfg.rng_key

    model_name = model_config.model_name
    dim = model_config.dim
    run_evaluation = model_config.run_evaluation
    num_samples = sampler_config.num_samples
    num_chains = sampler_config.num_chains
    burnin = sampler_config.burnin
    thinning = sampler_config.thinning
    sampler_type = sampler_config.sampler_type
    step_size = sampler_config.step_size
    num_integration_steps = sampler_config.num_integration_steps
    metric_method = sampler_config.metric_method
    total_num_steps = burnin + num_samples * thinning

    rng_key = jr.key(seed)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    model, dim = set_model(model_name, dim)
    logdensity_fn = model.logp
    inverse_mass_matrix = jnp.ones(dim)
    alpha2 = 1.0
    metric_fn = set_metric_fn(model, metric_method)
    params = set_params_sampler(
        sampler_type,
        step_size,
        num_integration_steps,
        alpha2,
        inverse_mass_matrix,
        metric_fn,
    )
    sampler = set_sampler(logdensity_fn, sampler_type, params)

    initial_positions = jnp.zeros((num_chains, dim))
    initial_states = jax.vmap(sampler.init)(initial_positions)
    states, elapsed_time = inference_loop_multiple_chains(
        rng_key, sampler, initial_states, total_num_steps, num_chains
    )
    samples = states.position
    samples = samples[burnin::thinning]
    samples = samples.reshape(num_samples * num_chains, dim)

    if dim == 2:
        plot_samples_marginal(samples, model, file_name=f"{output_dir}/samples.png")

    if run_evaluation:
        true_samples = get_reference_draws(model, model_name)
        distances1, distances2 = evaluate(rng_key, samples, true_samples, repeats)
        np.save(f"{output_dir}/distances1.npy", distances1)
        np.save(f"{output_dir}/distances2.npy", distances2)
        del distances1, distances2
        del true_samples

    del samples
    gc.collect()


if __name__ == "__main__":
    my_app()
