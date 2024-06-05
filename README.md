# Experiments

Contains the code used to manage the experiments for Geometric NUTS.

- Install requierements.txt

For posteriordb models, the github repository https://github.com/stan-dev/posteriordb/tree/master most be cloneed on the correct directory.

Run `sample.py` and select the model and the sampler.

```sh
python sample.py model=funnel sampler=nuts model.run_evaluation=False
```

For benchmarking reference samples are need. Install the modules: `pystan`, `cmdstanpy`.
Run the following to obtain the reference samples,
```sh
python expgeomjax/get_reference_samples/logreg/get_reference_samples.py
python expgeomjax/get_reference_samples/posteriordb_models/get_reference_samples.py
```


