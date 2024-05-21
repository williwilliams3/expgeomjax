"""

 -  Get true samples given by posteriordb package
    - make unconstrained (transform to real line) and save csv

"""

import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), os.pardir)
)
sys.path.append(PROJECT_ROOT)
print(PROJECT_ROOT)

from posteriordb import PosteriorDatabase
import pandas as pd
import arviz as az
import numpy as np
import stan


def get_posterior(name, pdb_path="posteriordb/posterior_database"):
    try:
        my_pdb = PosteriorDatabase(pdb_path)
        # my_pdb.posterior_names() # names of posteriors
        posterior = my_pdb.posterior(name)
        return posterior
    except:
        raise Exception("Unable to load posteriordb model")


def get_referencedraws(name, posterior, print_summary=True):
    gs = posterior.reference_draws()
    # num_params = sum(posterior.posterior_info["dimensions"].values())
    gs = pd.DataFrame(gs)
    gs["chain"] = range(10)

    df = gs.explode(gs.columns[:-1].to_list())
    df["draw"] = np.tile(range(1000), 10)
    df[gs.columns[:-1]] = df[gs.columns[:-1]].astype(float)
    samples_xa = df.set_index(["chain", "draw"]).to_xarray()
    if print_summary:
        res = az.summary(samples_xa)
        print(res)
    return df


def make_unconstrained_param(model_stan, row):
    constrained_param_dict = {}
    dim_list_cs = np.cumsum([0] + [1 if not dim else dim[0] for dim in model_stan.dims])

    for param_name, dim_start, dim_end in zip(
        model_stan.param_names, dim_list_cs[:-1], dim_list_cs[1:]
    ):
        constrained_param_dict[param_name] = row[dim_start:dim_end].tolist()

    return model_stan.unconstrain_pars(constrained_param_dict)


def constrained_to_unconstrained_params(posterior, samples):
    model_pdb, data_pdb = posterior.model, posterior.data
    model_stan = stan.build(
        model_pdb.stan_code(), data=data_pdb.values(), random_seed=42
    )
    return np.apply_along_axis(
        lambda row: make_unconstrained_param(model_stan, row), 1, samples
    )


if __name__ == "__main__":
    model_category = "reference_samples"
    model_names = [
        "arK-arK",
        "arma-arma11",
        "dogs-dogs",
        "dogs-dogs_log",
        "earnings-logearn_interaction",
        "eight_schools-eight_schools_centered",
        "eight_schools-eight_schools_noncentered",
        "garch-garch11",
        "gp_pois_regr-gp_regr",
        "hudson_lynx_hare-lotka_volterra",
        "low_dim_gauss_mix-low_dim_gauss_mix",
        "nes2000-nes",
        "sblrc-blr",
    ]

    for model_name in model_names:

        path = f"{model_category}/{model_name}"
        posterior = get_posterior(model_name)
        d3 = get_referencedraws(model_name, posterior)
        d3_unconstrained = constrained_to_unconstrained_params(posterior, d3)
        if not os.path.exists(f"data/{path}"):
            os.makedirs(f"data/{path}")
        # Save samples as csv
        df = pd.DataFrame(d3_unconstrained)
        df["chain"] = d3.reset_index().chain
        df["draw"] = d3.reset_index().draw
        df.to_csv(f"data/{path}/reference_samples.csv", index=False)
