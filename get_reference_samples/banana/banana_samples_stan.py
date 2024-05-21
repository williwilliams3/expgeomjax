import os
import postjax

import numpy as np
from cmdstanpy import CmdStanModel

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))


def generate_samples_stan(
    M,
    seed=1,
    N=10_000,
    show_progress=False,
    samples_dir=f"data/ground_truth_samples/banana/",
):
    data = dict(y=M.data.tolist())
    model = CmdStanModel(
        stan_file=os.path.join(current_directory, "stan_models/banana.stan")
    )
    fit = model.sample(
        data=data,
        thin=10,
        chains=10,
        iter_warmup=10_000,
        iter_sampling=N,
        adapt_delta=0.95,
        seed=seed,
        show_progress=show_progress,
    )
    samples = fit.draws_pd()[["theta[1]", "theta[2]"]].to_numpy()
    if samples_dir is not None:
        os.makedirs(samples_dir, exist_ok=True)
        np.save(samples_dir + "reference_samples.npy", samples)
    return samples


if __name__ == "__main__":
    M = postjax.banana()

    samples = generate_samples_stan(
        M,
        samples_dir=os.path.join(current_directory, "data/reference_samples/banana/"),
    )
