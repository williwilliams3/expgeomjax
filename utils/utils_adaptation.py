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
        (state, parameters), _ = adaptation_algo.run(rng_key, position)
        extra_parameters["inverse_mass_matrix"] = parameters["inverse_mass_matrix"]
        position = state.position
        # Adapt step size
        adaptation_algo = geomjax.step_size_adaptation(
            sampler_fn,
            logdensity_fn,
            **extra_parameters,
        )
    elif sampler_type in ["nutslmcmonge"]:
        # Get inverse mass matrix
        adaptation_algo = geomjax.window_adaptation(
            geomjax.nuts,
            logdensity_fn,
            is_mass_matrix_diagonal=True,
        )
        (state, parameters), _ = adaptation_algo.run(rng_key, position)
        extra_parameters["inverse_mass_matrix"] = parameters["inverse_mass_matrix"]
        position = state.position
        # Adapt step size
        adaptation_algo = geomjax.step_size_adaptation(
            sampler_fn,
            logdensity_fn,
            **extra_parameters,
        )
    else:
        # Adapt step size
        adaptation_algo = geomjax.step_size_adaptation(
            sampler_fn, logdensity_fn, **extra_parameters
        )
    (state, parameters), info = adaptation_algo.run(rng_key, position)
    if sampler_type in ["lmcmonge", "nutslmcmonge", "nutslmcmongeid"]:
        parameters = {**parameters, **extra_parameters}
    return (state, parameters), info


def adaptation_chees(
    sampler_type,
    logdensity_fn,
    rng_key,
    positions,
    num_chains,
    initial_step_size,
    inverse_mass_matrix=None,
    metric_fn=None,
):
    learning_rate = 0.025
    optim = optax.adam(learning_rate)
    if sampler_type == "cheeshmc":
        warmup = geomjax.chees_adaptation(logdensity_fn, num_chains)

    if sampler_type == "cheeslmc":
        warmup = geomjax.chees_adaptation_riemanian(
            logprob_fn=logdensity_fn,
            num_chains=num_chains,
            metric_fn=metric_fn,
            dynamics="lmc",
        )

    elif sampler_type == "cheesrmhmc":
        warmup = geomjax.chees_adaptation_riemanian(
            logprob_fn=logdensity_fn,
            num_chains=num_chains,
            metric_fn=metric_fn,
            dynamics="rmhmc",
        )
    elif sampler_type == "cheeslmcmonge":
        warmup = geomjax.chees_adaptation_lmcmonge(
            logprob_fn=logdensity_fn,
            num_chains=num_chains,
            inverse_mass_matrix=inverse_mass_matrix,
            update_alpha=False,
        )

    (last_states, parameters), info = warmup.run(
        rng_key, positions, initial_step_size, optim
    )
    return (last_states, parameters), info
