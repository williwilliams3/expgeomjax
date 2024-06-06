# Experiments

Contains the code used to manage the experiments for Geometric NUTS.

- Install requierements.txt

For posteriordb models, the github repository https://github.com/stan-dev/posteriordb/tree/master most at the same level as the current repository.

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

The complete set of results for the experiments, can be downloaded as .csv files from [here](https://www.dropbox.com/scl/fo/q3gnvwctvny3izjj7waw2/AJIhYQcpt_jbdmGRnI-_Mfc?rlkey=8xoo8ihumbuefxa6lblpres5d&st=6hk4zzk8&dl=0)
