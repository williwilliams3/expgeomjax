import postjax


def set_model(name_model, dim):
    if name_model == "funnel":
        M = postjax.neal_funnel(D=dim)
    elif name_model == "squiggle":
        M = postjax.squiggle(D=dim)
    elif name_model == "rosenbrock":
        M = postjax.hybrid_rosenbrock(D=dim, n1=1)
    elif name_model == "banana":
        M = postjax.banana()
        dim = M.D
    elif name_model == "logreg_australian":
        M = postjax.baylogreg(dataset_name="australian")
        dim = M.D
    elif name_model == "logreg_german":
        M = postjax.baylogreg(dataset_name="german")
        dim = M.D
    elif name_model == "logreg_heart":
        M = postjax.baylogreg(dataset_name="heart")
        dim = M.D
    elif name_model == "logreg_pima":
        M = postjax.baylogreg(dataset_name="pima")
        dim = M.D
    elif name_model == "logreg_ripley":
        M = postjax.baylogreg(dataset_name="ripley")
        dim = M.D
    elif name_model == "arK":
        M = postjax.posteriordb_models.arK()
        dim = M.D
    elif name_model == "arma11":
        M = postjax.posteriordb_models.arma11()
        dim = M.D
    elif name_model == "dogs":
        M = postjax.posteriordb_models.dogs_dogs()
        dim = M.D
    elif name_model == "dogs_log":
        M = postjax.posteriordb_models.dogs_log()
        dim = M.D
    elif name_model == "logearn_interaction":
        M = postjax.posteriordb_models.logearn_interaction()
        dim = M.D
    elif name_model == "eight_schools_centered":
        M = postjax.posteriordb_models.eight_schools_centered()
        dim = M.D
    elif name_model == "eight_schools_noncentered":
        M = postjax.posteriordb_models.eight_schools_noncentered()
        dim = M.D
    elif name_model == "garch11":
        M = postjax.posteriordb_models.garch11()
        dim = M.D
    elif name_model == "gp_regr":
        M = postjax.posteriordb_models.gp_regr()
        dim = M.D
    elif name_model == "low_dim_gauss_mix":
        M = postjax.posteriordb_models.low_dim_gauss_mix()
        dim = M.D
    elif name_model == "nes":
        M = postjax.posteriordb_models.nes()
        dim = M.D
    elif name_model == "blr":
        M = postjax.posteriordb_models.blr()
        dim = M.D
    elif name_model == "TwoMoons":
        M = postjax.twomoons()
        dim = M.D
    else:
        raise ValueError(f"Model {name_model} not found")
    return M, dim


def set_list_models(models: str):
    if models == "toy_models":
        name_model_list = [
            "funnel",
            "rosenbrock",
            "squiggle",
        ]
    elif models == "posteriordb":
        name_model_list = [
            "arK",
            "arma11",
            "dogs",
            # "dogs_log", # bug https://github.com/stan-dev/posteriordb/issues/245
            "logearn_interaction",
            "eight_schools_centered",
            "eight_schools_noncentered",
            "garch11",
            "gp_regr",
            "low_dim_gauss_mix",
            "nes",
            "blr",
        ]
    elif models == "logreg":
        name_model_list = [
            "logreg_australian",
            "logreg_german",
            "logreg_heart",
            "logreg_pima",
            "logreg_ripley",
        ]
    else:  # assume it is a single name
        return [models]
    return name_model_list
