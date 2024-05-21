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
            os.path.join(samples_dir, f"reference_samples_{self.dataset_name}.npy"),
            samples,
        )

    return samples


if __name__ == "__main__":
    from postjax.bayesian_log_reg import baylogreg

    M = baylogreg(dataset_name="australian")
    samples = generate_samples_stan(
        M,
        samples_dir=os.path.join(
            current_directory, "data/ground_truth_samples/banana/"
        ),
    )
    print(samples.shape)

    M = baylogreg(
        dataset_name="german",
        samples_dir=os.path.join(current_directory, "data/reference_samples/banana/"),
    )
    generate_samples_stan(M)

    M = baylogreg(
        dataset_name="heart",
        samples_dir=os.path.join(current_directory, "data/reference_samples/banana/"),
    )
    generate_samples_stan(M)

    M = baylogreg(
        dataset_name="pima",
        samples_dir=os.path.join(current_directory, "data/reference_samples/banana/"),
    )
    generate_samples_stan(M)

    M = baylogreg(
        dataset_name="ripley",
        samples_dir=os.path.join(current_directory, "data/reference_samples/banana/"),
    )
    generate_samples_stan(M)
