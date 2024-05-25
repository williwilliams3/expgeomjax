import geomjax


def set_sampler(sampler_type):
    if sampler_type == "hmc":
        sampler = geomjax.hmc
    elif sampler_type == "lmc":
        sampler = geomjax.lmc
    elif sampler_type in ["lmcmonge", "lmcmongeid"]:
        sampler = geomjax.lmcmonge
    elif sampler_type == "rmhmc":
        sampler = geomjax.rmhmc
    elif sampler_type == "nuts":
        sampler = geomjax.nuts
    elif sampler_type == "nutslmc":
        sampler = geomjax.nutslmc
    elif sampler_type in ["nutslmcmonge", "nutslmcmongeid"]:
        sampler = geomjax.nutslmcmonge
    elif sampler_type == "nutsrmhmc":
        sampler = geomjax.nutsrmhmc
    elif sampler_type == "cheeshmc":
        sampler = geomjax.dynamic_hmc
    elif sampler_type == "cheeslmc":
        sampler = geomjax.dynamic_lmc
    elif sampler_type == "cheesrmhmc":
        sampler = geomjax.dynamic_rmhmc
    elif sampler_type == "cheeslmcmonge":
        sampler = geomjax.dynamic_lmcmonge
    else:
        raise ValueError("Invalid sampler type")
    return sampler


def set_params_sampler(
    sampler_type,
    step_size,
    num_integration_steps,
    alpha2,
    inverse_mass_matrix,
    metric_fn,
    stopping_criterion,
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
    elif sampler_type in ["lmcmonge", "lmcmongeid"]:
        params["step_size"] = step_size
        params["num_integration_steps"] = num_integration_steps
        params["alpha2"] = alpha2
        params["inverse_mass_matrix"] = inverse_mass_matrix
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
        params["stopping_criterion"] = stopping_criterion
    elif sampler_type == "nutsrmhmc":
        params["step_size"] = step_size
        params["metric_fn"] = metric_fn
        params["stopping_criterion"] = stopping_criterion
    elif sampler_type in ["nutslmcmonge", "nutslmcmongeid"]:
        params["step_size"] = step_size
        params["alpha2"] = alpha2
        params["inverse_mass_matrix"] = inverse_mass_matrix
        params["stopping_criterion"] = stopping_criterion
    elif sampler_type == "cheeshmc":
        params["step_size"] = step_size
        params["inverse_mass_matrix"] = inverse_mass_matrix
    elif sampler_type in ["cheeslmc", "cheesrmhmc", "cheeslmcmonge"]:
        params["step_size"] = step_size
    else:
        raise ValueError("Invalid sampler type")
    return params
