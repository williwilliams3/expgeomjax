import os
from cmdstanpy import CmdStanModel
import numpy as np

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))


def generate_samples_stan(
    self,
    seed=1,
    N=10_000,
    show_progress=False,
    samples_dir=f"data/ground_truth_samples/logreg",
):

    data = {}
    data["N"] = self.data["N"]
    data["D"] = self.data["D"]
    data["y"] = self.data["y"].astype(int).tolist()
    data["x"] = self.data["X"].tolist()

    model = CmdStanModel(
        stan_file=os.path.join(current_directory, "stan_models/lr.stan")
    )

    # default options from posteriordb
    fit = model.sample(
        data=data,
        thin=10,
        chains=10,
        iter_warmup=10000,
        iter_sampling=N,
        seed=seed,
        show_progress=show_progress,
    )
    print(fit.summary())

    param_names = [f"beta[{n + 1}]" for n in range(self.D)]
    samples = fit.draws_pd()[param_names].to_numpy()

    if samples_dir is not None:
        os.makedirs(samples_dir, exist_ok=True)
        np.save(
            os.path.join(samples_dir, f"reference_samples.npy"),
            samples,
        )

    return samples


if __name__ == "__main__":
    from postjax.bayesian_log_reg import baylogreg

    for dataset_name in ["australian", "german", "heart", "pima", "ripley"]:
        M = baylogreg(dataset_name=dataset_name)
        samples = generate_samples_stan(
            M,
            samples_dir=os.path.join(
                current_directory, f"data/reference_samples/logreg/{dataset_name}"
            ),
        )
