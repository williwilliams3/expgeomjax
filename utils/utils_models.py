import postjax


def set_model(name_model, dim, sub_model=""):
    if name_model == "funnel":
        M = postjax.neal_funnel(D=dim)
    elif name_model == "squiggle":
        M = postjax.squiggle(D=dim)
    elif name_model == "rosenbrock":
        M = postjax.hybrid_rosenbrock(D=dim, n1=1)
    elif name_model == "banana":
        M = postjax.banana()
        dim = M.D
    elif name_model == "logreg":
        M = postjax.baylogreg(dataset_name=sub_model)
        dim = M.D
    elif name_model == "postdb":
        if sub_model == "arK-arK" in sub_model:
            M = postjax.posteriordb_models.arK()
            dim = M.D
        elif sub_model == "arma-arma11":
            M = postjax.posteriordb_models.arma11()
            dim = M.D
        elif sub_model == "dogs-dogs":
            M = postjax.posteriordb_models.dogs_dogs()
            dim = M.D
        elif sub_model == "earnings-logearn_interaction":
            M = postjax.posteriordb_models.logearn_interaction()
            dim = M.D
        elif sub_model == "eight_schools-eight_schools_centered":
            M = postjax.posteriordb_models.eight_schools_centered()
            dim = M.D
        elif sub_model == "eight_schools-eight_schools_noncentered":
            M = postjax.posteriordb_models.eight_schools_noncentered()
            dim = M.D
        elif sub_model == "garch-garch11":
            M = postjax.posteriordb_models.garch11()
            dim = M.D
        elif sub_model == "gp_pois_regr-gp_regr":
            M = postjax.posteriordb_models.gp_regr()
            dim = M.D
        elif sub_model == "low_dim_gauss_mix-low_dim_gauss_mix":
            M = postjax.posteriordb_models.low_dim_gauss_mix()
            dim = M.D
        elif sub_model == "nes2000-nes":
            M = postjax.posteriordb_models.nes()
            dim = M.D
        elif sub_model == "sblrc-blr":
            M = postjax.posteriordb_models.blr()
            dim = M.D
    elif name_model == "twomoons":
        M = postjax.twomoons()
        dim = M.D
    else:
        raise ValueError(f"Model {name_model} not found")
    return M, dim
