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
from utils.utils_evaluate import number_gradient_evaluations, estimate_implicit_steps
from utils.utils_sampling import inference_loop_multiple_chains_pmap
from utils.utils_adaptation import adaptation_chees
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
    make_plots = model_config.make_plots
    if model_name in ["logreg", "postdb"]:
        sub_name = model_config.sub_name
    else:
        sub_name = ""

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
    if sampler_type in ["nutslmc", "nutslmcmonge", "nutsrmhmc", "nutslmcmongeid"]:
        stopping_criterion = sampler_config.stopping_criterion
    else:
        stopping_criterion = None
    if metric_method == "softabs":
        alpha = 1e6

    total_num_steps = burnin + (num_samples // num_chains) * thinning

    rng_key = jr.key(seed)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    model, dim = set_model(model_name, dim, sub_name)
    logdensity_fn = model.logp

    # Initial Position (zero vector)
    position = jnp.zeros(dim)
    positions = jnp.zeros((num_chains, dim))

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
            stopping_criterion,
        )
        if sampler_type in ["lmcmonge", "nutslmcmonge", "lmcmongeid", "nutslmcmongeid"]:
            extra_params = {
                key: params[key] for key in params if key not in ["step_size"]
            }
        else:
            extra_params = {
                key: params[key]
                for key in params
                if key not in ["step_size", "inverse_mass_matrix"]
            }

        if "chees" in sampler_type:
            (states, params), info_adapt = adaptation_chees(
                sampler_type,
                logdensity_fn,
                rng_key,
                positions,
                num_chains,
                step_size,
                inverse_mass_matrix=inverse_mass_matrix,
                metric_fn=metric_fn,
            )
        else:
            (state, params), info_adapt = adaptation(
                sampler_fn, sampler_type, logdensity_fn, rng_key, position, extra_params
            )
        print("Adapted parameters:", params)
        sampler = sampler_fn(logdensity_fn, **params)
        if "chees" not in sampler_type:
            states = jax.vmap(sampler.init)(jnp.tile(state.position, (num_chains, 1)))

    else:

        params = set_params_sampler(
            sampler_type,
            step_size,
            num_integration_steps,
            alpha,
            inverse_mass_matrix,
            metric_fn,
            stopping_criterion,
        )
        sampler = sampler_fn(logdensity_fn, **params)
        states = jax.vmap(sampler.init)(positions)

    if "nuts" in sampler_type:
        states, info, elapsed_time = inference_loop_multiple_chains_pmap(
            rng_key, sampler, states, total_num_steps, num_chains
        )
        samples_tensor = states.position.transpose((1, 0, 2))
    else:
        states, info, elapsed_time = inference_loop_multiple_chains(
            rng_key, sampler, states, total_num_steps, num_chains
        )
        samples_tensor = states.position
    print(f"MCMC elapsed time: {elapsed_time:.2f} seconds")
    print(f"Acceptance rate: {info.acceptance_rate.mean()}")

    samples_tensor = samples_tensor[burnin::thinning]
    samples = samples_tensor.reshape(num_samples, dim)

    if run_evaluation:
        # ESS and Rhat
        rhat = geomjax.rhat(samples_tensor, chain_axis=1, sample_axis=0)
        ess = geomjax.ess(samples_tensor, chain_axis=1, sample_axis=0)
        if "rmhmc" in sampler_type:
            momentums = info.momentum
            if "nuts" in sampler_type:
                momentums = momentums.transpose((1, 0, 2))
            momentums = momentums[burnin::thinning]
            momentums = momentums.reshape(num_samples, dim)
            average_implicit_steps = estimate_implicit_steps(
                samples, momentums, step_size, logdensity_fn, metric_fn
            )
        else:
            average_implicit_steps = 1
        print(f"Rhat: {rhat}")
        print(f"ESS: {ess}")
        if "rmhmc" in sampler_type:
            print(f"Average implicit steps: {average_implicit_steps}")
        gradient_evaluations = number_gradient_evaluations(
            sampler_type,
            total_num_steps,
            num_chains,
            num_integration_steps,
            info,
            average_implicit_steps=average_implicit_steps,
        )

        sampling_stats = {
            "rhat": rhat.tolist(),
            "ess": ess.tolist(),
            "elapsed_time": float(elapsed_time),
            "average_acceptance_rate": float(info.acceptance_rate.mean()),
            "gradient_evaluations": int(gradient_evaluations),
        }
        with open(f"{output_dir}/stats.json", "w") as f:
            json.dump(sampling_stats, f)

        # Wasserstein distance

        true_samples = get_reference_draws(model, model_name, num_samples, sub_name)
        distances1, distances2 = evaluate(rng_key, samples, true_samples, repeats)
        np.save(f"{output_dir}/distances1.npy", distances1)
        np.save(f"{output_dir}/distances2.npy", distances2)
        if model_name in ["funnel", "rosenbrock", "squiggle"]:
            col_index = -1 if model_name == "funnel" else 0
            distances_marginal1, distances_marginal2 = evaluate(
                rng_key,
                samples[:, col_index : jnp.newaxis],
                true_samples[:, col_index : jnp.newaxis],
                repeats,
            )
            np.save(f"{output_dir}/distances_marginal1.npy", distances_marginal1)
            np.save(f"{output_dir}/distances_marginal2.npy", distances_marginal2)
            del distances_marginal1, distances_marginal2

        del distances1, distances2
        del true_samples

    if make_plots:
        # Plots first and last dimensions
        remove_outliers_theta0 = model_name == "funnel"
        if sampler_type == "nuts":
            scatter_color = "#FF7F7F"
        else:
            scatter_color = "#7FD6FF"
        plot_samples_marginal(
            samples,
            model,
            file_name=f"{output_dir}/samples.png",
            remove_outliers_theta0=remove_outliers_theta0,
            scatter_color=scatter_color,
        )

    del samples, samples_tensor
    gc.collect()


if __name__ == "__main__":
    my_app()
