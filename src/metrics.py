import jax.numpy as jnp

# Taken from
# https://github.com/JamesBrofos/Iliad/blob/main/iliad/integrators/fields/softabs.py


def eigen_to_matrix(eigen: jnp.ndarray, vectors: jnp.ndarray) -> jnp.ndarray:
    """Computes the vector whose eigen-decomposition is provided. Assumes that the
    eigenvectors are orthonormal.

    Args:
        eigen: Eigenvalues of the matrix.
        vectors: Eigenvectors of the matrix.

    Returns:
        matrix: The matrix with the provided eigen-decomposition.

    """
    matrix = jnp.dot(vectors, (eigen * vectors).T)
    matrix = jnp.real(matrix)
    return matrix


def coth(x: jnp.ndarray) -> jnp.ndarray:
    """Implements the hyperbolic cotangent function."""
    # out = jnp.where(jnp.abs(x) > 1e-8, jnp.reciprocal(jnp.tanh(x)), 0.0)
    out = jnp.reciprocal(jnp.tanh(x))
    return out


def softabs(lam: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """The softabs transformation that applies a smooth absolute value-like
    transformation to the input. The sharpness of the transformation about zero
    is controlled by the parameter `alpha`.

    Args:
        lam: Values to smoothly transform under the softabs operation.
        alpha: The softabs smoothness parameter.

    Returns:
        slam: The transformed values under the softabs operation.

    """
    slam = lam * coth(alpha * lam)
    return slam


def softabs_metric(H, alpha):
    l, U = jnp.linalg.eigh(H)
    lt = softabs(l, alpha)
    metric = eigen_to_matrix(lt, U)
    return metric
