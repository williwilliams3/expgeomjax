import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import jax


def sub_sample(samples, max_samples=1000):
    n_rows, n_cols = samples.shape
    if n_rows <= max_samples:
        return samples
    if n_rows > max_samples:
        id_samples = np.random.choice(n_rows, size=max_samples, replace=False)
        return samples[id_samples]


def plot_samples_marginal(
    samples,
    M,
    densities=True,
    xaxes=["θ1", "θ2"],
    save_fig=True,
    scatter_color="#7FD6FF",
    file_name="figs/samples.png",
    remove_outliers_theta0=False,
    do_use_latex=False,
):
    hist_edge_color = "#D5D8DC"
    hist_face_color = scatter_color
    density_color = "#273746"
    cmap_colors = plt.cm.get_cmap("Greys").reversed()

    if do_use_latex == True:
        plt.rcParams["text.usetex"] = True
        # https://stackoverflow.com/a/14324826, in order to use \boldsymbol
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
        xaxes = [r"$\boldsymbol{\theta}_{1}$", r"$\boldsymbol{\theta}_{2}$"]

    D_old = M.D
    xlim = M.xlim
    ylim = M.ylim
    if M.D > 2:
        M.__init__()

    g = lambda x, y: np.exp(M.logp(np.array([x, y])))
    # g = lambda x,y: M.logp([x,y])
    if densities:
        density1, density2 = M.densities()
    # get only first and last column
    samples = samples[:, [0, -1]]
    sub_samples = sub_sample(samples, max_samples=4000)

    if M.name == "banana":
        true_dist_levels = [-210.0, -205.0, -204.0]
        contours = get_contours(xlim, ylim, M.logp)
        [X, Y, Z] = contours
        levels = true_dist_levels
    else:
        # Plot everything together
        x0 = np.arange(xlim[0], xlim[1], 0.15)
        x1 = np.arange(ylim[0], ylim[1], 0.15)
        X, Y = np.meshgrid(x0, x1)
        g = np.vectorize(g)
        Z = g(X, Y)
        levels = 20

    plt.rcParams.update({"font.size": 22})
    fig, axs = plt.subplots(
        2,
        2,
        figsize=(6, 6),
        gridspec_kw={"width_ratios": [2, 0.5], "height_ratios": [0.5, 2.0]},
    )
    # Top left histogram
    if remove_outliers_theta0:
        filtered = samples[:, 0][~is_outlier(samples[:, 0])]
    else:
        filtered = samples[:, 0]
    axs[0, 0].hist(
        filtered,
        bins=40,
        density=True,
        edgecolor=hist_edge_color,
        facecolor=hist_face_color,
        label="Samples",
        alpha=0.8,
    )
    if density1 is not None:
        axs[0, 0].plot(x0, density1(x0), label="True", color=density_color)
    axs[0, 0].set_xticklabels([])
    axs[0, 0].set_yticklabels([])
    axs[0, 0].legend(loc="upper right", prop={"size": 12})
    axs[0, 0].set_xlim(xlim[0], xlim[1])
    axs[0, 0].spines["top"].set_visible(False)
    axs[0, 0].spines["right"].set_visible(False)
    # axs[0].spines["bottom"].set_visible(False)
    axs[0, 0].spines["left"].set_visible(False)

    # Contour plot bottom left
    axs[1, 0].contour(X, Y, Z, levels=levels, cmap=cmap_colors, linestyles="dashed")
    axs[1, 0].set_xlabel(xaxes[0])
    axs[1, 0].set_ylabel(xaxes[1])
    axs[1, 0].set_xlim(xlim[0], xlim[1])
    axs[1, 0].set_ylim(ylim[0], ylim[1])
    axs[1, 0].scatter(
        sub_samples[:, 0],
        sub_samples[:, 1],
        alpha=0.15,
        s=20,
        marker="o",
        zorder=2,
        color=scatter_color,
    )

    # bottom right histogram
    axs[1, 1].hist(
        samples[:, 1],
        bins=40,
        density=True,
        edgecolor=hist_edge_color,
        facecolor=hist_face_color,
        label="Samples",
        orientation="horizontal",
        alpha=0.8,
    )
    if density2 is not None:
        axs[1, 1].plot(density2(x1), x1, label="True", color=density_color)
    axs[1, 1].set_xticklabels([])
    axs[1, 1].set_yticklabels([])
    # axs[1, 1].legend(loc="upper right", prop={"size": 12})
    axs[1, 1].set_ylim(ylim[0], ylim[1])
    axs[1, 1].spines["top"].set_visible(False)
    axs[1, 1].spines["right"].set_visible(False)
    axs[1, 1].spines["bottom"].set_visible(False)
    # axs[1].spines["left"].set_visible(False)

    # make last one invisible
    axs[0, 1].axis("off")

    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    if save_fig:
        plt.savefig(file_name, dpi=200, bbox_inches="tight")
        print("Plot saved")
    else:
        plt.show()
    # return old value
    M.__init__(D=D_old)


def cum_mean(arr):
    cum_sum = np.cumsum(arr, axis=0)
    for i in range(cum_sum.shape[0]):
        if i == 0:
            continue
        # print(cum_sum[i] / (i + 1))
        cum_sum[i] = cum_sum[i] / (i + 1)
    return cum_sum


def plot_chains(samples, save_fig=True, file_name="figs/chains.png"):
    n_samples, n_chains, n_dim = samples.shape
    num_rows = (n_dim - 1) // 3 + 1

    fig, axes = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows), clear=True, num=1)
    fig.subplots_adjust(hspace=0.4)

    for i, ax in enumerate(axes.flat):
        if i < n_dim:
            for chain_idx in range(n_chains):
                x = np.arange(n_samples)  # x-axis values for the trajectories
                param_name = f"theta.{i}"
                ax.plot(
                    x,
                    samples[:, chain_idx, i],
                    label=f"Chain {chain_idx + 1}",
                    alpha=0.4,
                )
                ax.set_title(param_name)
            # ax.legend()
            ax.set_xlabel("Iteration")
            if i == 1:
                ax.set_ylabel("Chains")
        else:
            ax.axis("off")  # Hide unused subplot

    plt.tight_layout()

    if save_fig:
        plt.savefig(file_name, dpi=200)
        print("plot chains saved")
        fig.clear()
    else:
        plt.show()


def plot_histograms(samples, save_fig=True, file_name="figs/hist.png"):
    arr_dims = np.ndim(samples)
    n_dim = samples.shape[-1]
    num_rows = (n_dim - 1) // 3 + 1
    fig, axes = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows), clear=True, num=2)
    fig.subplots_adjust(hspace=0.4)

    for i, ax in enumerate(axes.flat):
        if i < n_dim:
            param_name = f"theta.{i}"
            if arr_dims == 2:
                samples_marginal = samples[:, i]
            elif arr_dims == 3:
                samples_marginal = samples[:, :, i].reshape(-1)
            else:
                raise ValueError("Samples should be 2D or 3D")
            ax.hist(samples_marginal, bins=30, edgecolor="black")
            ax.set_title(param_name)
        else:
            ax.axis("off")  # Hide unused subplot
    plt.tight_layout()
    if save_fig:
        plt.savefig(file_name, dpi=200)
        print("histogram saved")
        fig.clear()

    else:
        plt.show()


def is_outlier(points, thresh=3.5):
    """
    https://stackoverflow.com/questions/11882393/matplotlib-disregard-outliers-when-plotting
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def make_plot_loss(losses, file_name="figs/temp/losses.png"):
    if isinstance(losses, dict):
        epochs = np.fromiter(losses.keys(), dtype=float)
        loss_values = np.fromiter(losses.values(), dtype=float)
    else:
        epochs = np.arange(len(losses))
        loss_values = np.array(losses)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss_values, label="Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(file_name, dpi=200, bbox_inches="tight")
    print("Plot losses saved")


def plot_metrics(
    M,
    metric_fn,
    xlim,
    ylim,
    multiplier=0.2,
    figsize=(10, 10),
    file_name="figs/metrics.png",
    color_points="green",
    do_filter=False,
    do_use_latex=False,
):
    xs = np.linspace(xlim[0], xlim[1], 7)
    ys = np.linspace(ylim[0], ylim[1], 7)

    D_old = M.D
    if M.D > 2:
        # TODO: Fix bug
        M.__init__()

    if M.name == "Banana":
        true_dist_levels = [-210.0, -205.0, -204.0]
        contours = get_contours(xlim, ylim, M.logp)
        [X, Y, Z] = contours
        levels = true_dist_levels
    else:
        g = lambda x, y: M.logp(np.array([x, y]))

        # Plot everything together
        # x0 = np.arange(xlim, xlim[1], 0.15)
        # x1 = np.arange(ylim[0], ylim[1], 0.15)
        x0 = np.linspace(*xlim, 100)
        x1 = np.linspace(*ylim, 100)

        X, Y = np.meshgrid(x0, x1)
        # vec_logp_fn = jax.jit(jax.vmap(jax.vmap(logp_fn, in_axes=1), in_axes=1))
        # Z = vec_logp_fn(jnp.stack([X, Y]))
        g = np.vectorize(g)
        Z = g(X, Y)
        Z = np.exp(Z)
        levels = 8

    # ChatGPT

    if do_use_latex:
        plt.rcParams["font.size"] = 35
        plt.rcParams["text.usetex"] = True
        # https://stackoverflow.com/a/74136954
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
        xaxes = [r"$\boldsymbol{\theta}_{1}$", r"$\boldsymbol{\theta}_{2}$"]
    else:
        xaxes = xaxes = ["θ1", "θ2"]
    fig, ax = plt.subplots(figsize=figsize)
    cmap_colors = plt.get_cmap("Greys").reversed()
    plt.contour(
        X,
        Y,
        Z,
        levels=levels,
        cmap=cmap_colors,
        linestyles="dashed",
        linewidths=1.5,
        zorder=1,
    )

    if do_filter:
        X0, X1 = np.meshgrid(xs, ys)
        Z = g(X0, X1)
        filter_balls = Z > np.quantile(Z, 0.1)
    else:
        filter_balls = np.ones((len(xs), len(ys)), dtype=bool)

    if M.name == "Rosenbrock":
        x_values = np.linspace(*xlim, 16)
        y_values = x_values**2
        points_mean = np.column_stack((x_values, y_values))

        points = points_mean
        for point in points:
            ellipse = get_individual_ellipse(
                point, metric_fn, multiplier, color_points, D_old
            )
            ax.add_patch(ellipse)
    elif M.name == "Squiggle":
        x_values = np.linspace(*xlim, 50)
        y_values = -np.sin(M.a * x_values)
        points_mean = np.column_stack((x_values, y_values))
        points = points_mean

        for point in points:
            ellipse = get_individual_ellipse(
                point, metric_fn, multiplier, color_points, D_old
            )
            ax.add_patch(ellipse)
    elif M.name == "Banana":
        y_values = np.linspace(*ylim, 16)
        x_values = -(y_values**2) + 1.25
        points_mean = np.column_stack((x_values, y_values))
        points = points_mean
        for point in points:
            ellipse = get_individual_ellipse(
                point, metric_fn, multiplier, color_points, D_old
            )
            ax.add_patch(ellipse)

    else:
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                if filter_balls[i, j]:
                    #  D>2 matrices, keep first and last dimensions
                    point = jnp.asarray([x, y])
                    ellipse = get_individual_ellipse(
                        point, metric_fn, multiplier, color_points, D_old
                    )
                    ax.add_patch(ellipse)

    plt.xlabel(xaxes[0])
    plt.ylabel(xaxes[1])
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])

    # ChatGPT
    plt.xticks([])
    plt.yticks([])

    plt.savefig(file_name, dpi=200, bbox_inches="tight")
    print("Plot saved")

    # Back to old value
    if D_old > 2:
        M.__init__(D=D_old)


def get_individual_ellipse(point, metric_fn, multiplier, color_points, D_old):
    x, y = point
    point_extended = jnp.asarray([x] + [0] * (D_old - 2) + [y])
    metric = np.array(metric_fn(point_extended))
    ndims = np.ndim(metric)
    if ndims == 1:
        metric = np.diag(metric[[0, -1]])

    else:
        metric = metric[[0, 0, -1, -1], [0, -1, 0, -1]].reshape(2, 2)

    # print(metric)
    try:
        covariance_matrix = np.linalg.inv(metric)
    except:
        print("point", point)
        raise ValueError("Matrix is not invertible")

    # based on ChatGPT
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    max_eig = np.sqrt(np.max(eigenvalues))
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Compute the lengths of major and minor axes
    major_axis = multiplier * np.sqrt(eigenvalues[0]) / max_eig
    minor_axis = multiplier * np.sqrt(eigenvalues[1]) / max_eig
    ellipse = Ellipse(
        xy=point,
        width=major_axis,
        height=minor_axis,
        angle=np.degrees(np.arctan2(*eigenvectors[:, 0][::-1])),
        facecolor="none",
        edgecolor=color_points,
        linewidth=2.5,
        zorder=10,
    )
    return ellipse


def get_contours(xlim, ylim, logp_fn):
    x0 = jnp.arange(xlim[0], xlim[1], 0.1)
    x1 = jnp.arange(ylim[0], ylim[1], 0.1)
    X, Y = jnp.meshgrid(x0, x1)
    vec_logp_fn = jax.jit(jax.vmap(jax.vmap(logp_fn, in_axes=1), in_axes=1))
    Z = vec_logp_fn(jnp.stack([X, Y]))
    contours = [np.asarray(X), np.asarray(Y), np.asarray(Z)]
    return contours


def plot_connected_scatter(
    array1, array2, xlim, ylim, out_fig_dir_name, num_points=50, do_connect_dots=False
):

    plt.figure()

    plt.scatter(
        array2[:num_points, 0],
        array2[:num_points, -1],
        edgecolors="red",
        facecolors="none",
        alpha=0.5,
        label="New",
    )
    if do_connect_dots:
        plt.scatter(
            array1[:num_points, 0],
            array1[:num_points, -1],
            facecolors="none",
            edgecolors="blue",
            alpha=0.5,
            label="Old",
        )
        array1_reshaped = array1[:num_points, np.newaxis, :]
        array2_reshaped = array2[:num_points, np.newaxis, :]
        # Concatenate arrays along the third axis to create an array of line segments
        segments = np.concatenate([array1_reshaped, array2_reshaped], axis=2)
        # Reshape segments array to (10, 4) to flatten the dimensions
        segments = segments.reshape(-1, 4)
        # Connect dots of the arrays
        plt.plot(
            segments[:, [0, 2]].T,
            segments[:, [1, 3]].T,
            color="gray",
            alpha=0.5,
            linestyle="dotted",
        )
        plt.legend()
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.savefig(out_fig_dir_name)
