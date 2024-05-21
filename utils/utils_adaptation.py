import geomjax
import optax


def adaptation(
    sampler_fn, sampler_type, logdensity_fn, rng_key, position, extra_parameters
):

    if sampler_type in ["hmc", "nuts"]:
        # Adapt monge with euclidean metric
        adaptation_algo = geomjax.window_adaptation(
            sampler_fn, logdensity_fn, is_mass_matrix_diagonal=True, **extra_parameters
        )
    elif sampler_type in ["lmcmonge"]:
        # Adapt monge with euclidean metric
        adaptation_algo = geomjax.window_adaptation(
            geomjax.hmc,
            logdensity_fn,
            is_mass_matrix_diagonal=True,
            num_integration_steps=extra_parameters["num_integration_steps"],
        )
    elif sampler_type in ["nutslmcmonge"]:
        # Adapt monge with euclidean metric
        adaptation_algo = geomjax.window_adaptation(
            geomjax.nuts,
            logdensity_fn,
            is_mass_matrix_diagonal=True,
        )
    else:
        adaptation_algo = geomjax.step_size_adaptation(
            sampler_fn, logdensity_fn, **extra_parameters
        )
    (state, parameters), info = adaptation_algo.run(rng_key, position)
    if sampler_type in ["lmcmonge", "nutslmcmonge"]:
        parameters["alpha2"] = extra_parameters["alpha2"]
    return (state, parameters), info


def adaptation_chees(
    sampler_type,
    logdensity_fn,
    rng_key,
    positions,
    num_chains,
    initial_step_size,
    metric_fn=None,
):
    learning_rate = 0.025
    if sampler_type == "cheeshmc":
        warmup = geomjax.chees_adaptation(logdensity_fn, num_chains)
        optim = optax.adam(learning_rate)
        (last_states, parameters), info = warmup.run(
            rng_key, positions, initial_step_size, optim
        )
    if sampler_type == "cheeslmc":
        warmup = geomjax.chees_adaptation_lmc(
            logprob_fn=logdensity_fn, num_chains=num_chains, metric_fn=metric_fn
        )
        optim = optax.adam(learning_rate)
        (last_states, parameters), info = warmup.run(
            rng_key, positions, initial_step_size, optim
        )
    return (last_states, parameters), info
