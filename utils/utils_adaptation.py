import geomjax


def adaptation(
    sampler_fn, sampler_type, logdensity_fn, rng_key, position, extra_parameters
):

    if sampler_type in ["hmc", "nuts", "lmcmonge", "nutslmcmonge"]:
        adaptation_algo = geomjax.window_adaptation(
            sampler_fn, logdensity_fn, is_mass_matrix_diagonal=True, **extra_parameters
        )
    else:
        adaptation_algo = geomjax.step_size_adaptation(
            sampler_fn, logdensity_fn, **extra_parameters
        )
    (state, parameters), info = adaptation_algo.run(rng_key, position)
    return (state, parameters), info
