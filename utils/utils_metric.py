import jax
import jax.numpy as jnp
from utils.metrics import softabs_metric


def set_metric_fisher_fn(M, is_cholesky):
    if is_cholesky:
        metric_fn = M.fisher_metric_fn
    else:
        metric_fn = lambda theta: jnp.diag(M.fisher_metric_fn(theta))
    return metric_fn


def set_metric_empirical_fisher_fn(M, is_cholesky):
    if is_cholesky:
        metric_fn = M.empirical_fisher_metric_fn
    else:
        metric_fn = lambda theta: jnp.diag(M.empirical_fisher_metric_fn(theta))
    return metric_fn


def set_metric_soft_fn(M, is_cholesky, alpha=1.0):
    if is_cholesky:

        def metric_fn(theta):
            hessian_fn = jax.hessian(M.logp)
            H = hessian_fn(theta)
            metric = softabs_metric(H, alpha=alpha)
            return metric

    else:

        def metric_fn(theta):
            hessian_fn = jax.hessian(M.logp)
            H = hessian_fn(theta)
            metric = softabs_metric(H, alpha=alpha)
            metric = jnp.diag(metric)
            return metric

    return metric_fn


def get_monge_metric(M, is_cholesky, alpha2=1.0):
    logdensity_fn = M.logp
    grad_fn = jax.grad(logdensity_fn)

    if is_cholesky:

        def metric_fn(theta):
            metric = jnp.eye(M.D) + alpha2 * jnp.outer(grad_fn(theta), grad_fn(theta))
            return metric

    else:

        def metric_fn(theta):
            metric = jnp.ones(M.D) + alpha2 * (grad_fn(theta) * grad_fn(theta))
            return metric

    return metric_fn


def set_metric_fn(M, metric_method, alpha=1.0, is_cholesky=True):

    if metric_method == "fisher":
        metric_fn = set_metric_fisher_fn(M, is_cholesky)
    elif metric_method == "softabs":
        metric_fn = set_metric_soft_fn(M, is_cholesky, alpha=alpha)
    elif metric_method == "monge":
        metric_fn = get_monge_metric(M, is_cholesky, alpha2=alpha)
    elif metric_method == "emp_fisher":
        metric_fn = set_metric_empirical_fisher_fn(M, is_cholesky)
    else:
        metric_fn = None
    return metric_fn
