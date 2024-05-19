import geomjax


def set_sampler(logdensity_fn, sampler_type, params):
    if sampler_type == "hmc":
        sampler = geomjax.hmc(logdensity_fn, **params)
    elif sampler_type == "lmc":
        sampler = geomjax.lmc(logdensity_fn, **params)
    elif sampler_type == "lmcmonge":
        sampler = geomjax.lmcmonge(logdensity_fn, **params)
    elif sampler_type == "rmhmc":
        sampler = geomjax.rmhmc(logdensity_fn, **params)
    elif sampler_type == "nuts":
        sampler = geomjax.nuts(logdensity_fn, **params)
    elif sampler_type == "nuts":
        sampler = geomjax.nuts(logdensity_fn, **params)
    elif sampler_type == "nuts":
        sampler = geomjax.nuts(logdensity_fn, **params)
    return sampler


def set_params_sampler(
    sampler_type,
    step_size,
    num_integration_steps,
    alpha2,
    inverse_mass_matrix,
    metric_fn,
):
    params = {}
    if sampler_type == "hmc":
        params["step_size"] = step_size
        params["num_integration_steps"] = num_integration_steps
        params["inverse_mass_matrix"] = inverse_mass_matrix
    elif sampler_type == "lmc":
        params["step_size"] = step_size
        params["num_integration_steps"] = num_integration_steps
        params["metric_fn"] = metric_fn
    elif sampler_type == "lmcmonge":
        params["step_size"] = step_size
        params["num_integration_steps"] = num_integration_steps
        params["alpha2"] = alpha2
    elif sampler_type == "rmhmc":
        params["step_size"] = step_size
        params["num_integration_steps"] = num_integration_steps
        params["metric_fn"] = metric_fn
    elif sampler_type == "nuts":
        params["step_size"] = step_size
        params["inverse_mass_matrix"] = inverse_mass_matrix
    elif sampler_type == "nutslmc":
        params["step_size"] = step_size
        params["metric_fn"] = metric_fn
    elif sampler_type == "nutsrmhmc":
        params["step_size"] = step_size
        params["metric_fn"] = metric_fn
    elif sampler_type == "nutslmcmonge":
        params["step_size"] = step_size
        params["alpha2"] = alpha2
    return params
