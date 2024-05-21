from jax import config
import os

config.update("jax_enable_x64", True)
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import sys
import gc

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import numpy as np
import jax
import jax.random as jr
import jax.numpy as jnp
import geomjax
import hydra
from omegaconf import OmegaConf
from utils import (
    set_model,
    inference_loop_multiple_chains,
    get_reference_draws,
    set_sampler,
    set_params_sampler,
    set_metric_fn,
    evaluate,
    adaptation,
)
from plotting.plotting_functions import plot_samples_marginal
import json


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
    alpha = sampler_config.alpha
    run_adaptation = sampler_config.run_adaptation

    total_num_steps = burnin + num_samples * thinning

    rng_key = jr.key(seed)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    model, dim = set_model(model_name, dim)
    logdensity_fn = model.logp
    position = jnp.zeros(dim)

    metric_fn = set_metric_fn(model, metric_method)
    inverse_mass_matrix = jnp.ones(dim)
    sampler_fn = set_sampler(sampler_type)
    if run_adaptation:
        params = set_params_sampler(
            sampler_type,
            step_size,
            num_integration_steps,
            alpha,
            inverse_mass_matrix,
            metric_fn,
        )
        extra_params = {
            key: params[key]
            for key in params
            if key not in ["step_size", "inverse_mass_matrix"]
        }
        (state, params), info_adapt = adaptation(
            sampler_fn, sampler_type, logdensity_fn, rng_key, position, extra_params
        )
        print("Adapted parameters:", params)
        sampler = sampler_fn(logdensity_fn, **params)
        states = jax.vmap(sampler.init)(jnp.tile(state.position, (num_chains, 1)))

    else:
        sampler = sampler_fn(logdensity_fn, **params)
        initial_positions = jnp.zeros((num_chains, dim))
        params = set_params_sampler(
            sampler_type,
            step_size,
            num_integration_steps,
            alpha,
            inverse_mass_matrix,
            metric_fn,
        )
        states = jax.vmap(sampler.init)(initial_positions)

    states, info, elapsed_time = inference_loop_multiple_chains(
        rng_key, sampler, states, total_num_steps, num_chains
    )
    print(f"MCMC elapsed time: {elapsed_time:.2f} seconds")
    print(f"Acceptance rate: {info.acceptance_rate.mean()}")
    samples_tensor = states.position
    samples_tensor = samples_tensor[burnin::thinning]
    samples = samples_tensor.reshape(num_samples * num_chains, dim)

    if dim == 2:
        remove_outliers_theta0 = model.name == "NealFunnel"
        plot_samples_marginal(
            samples,
            model,
            file_name=f"{output_dir}/samples.png",
            remove_outliers_theta0=remove_outliers_theta0,
        )

    if run_evaluation:
        # ESS and Rhat
        rhat = geomjax.rhat(samples_tensor, chain_axis=1, sample_axis=0)
        ess = geomjax.ess(samples_tensor, chain_axis=1, sample_axis=0)
        elapsed_time
        # TODO: Fix this
        # gradient_evaluations = number_gradient_evaluations()
        gradient_evaluations = total_num_steps * num_chains * num_integration_steps
        sampling_stats = {
            "rhat": float(rhat),
            "ess": float(ess),
            "elapsed_time": float(elapsed_time),
            "average_acceptance_rate": float(info.acceptance_rate.mean()),
            "gradient_evaluations": int(gradient_evaluations),
        }
        with open("stats.json", "w") as f:
            json.dump(sampling_stats, f)

        # Wasserstein distance
        true_samples = get_reference_draws(model, model_name)
        distances1, distances2 = evaluate(rng_key, samples, true_samples, repeats)
        np.save(f"{output_dir}/distances1.npy", distances1)
        np.save(f"{output_dir}/distances2.npy", distances2)
        del distances1, distances2
        del true_samples

    del samples, samples_tensor
    gc.collect()


if __name__ == "__main__":
    my_app()
