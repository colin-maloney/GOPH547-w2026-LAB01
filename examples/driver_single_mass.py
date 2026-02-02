import numpy as np
import matplotlib.pyplot as plt 
import sys, pathlib 
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.goph547lab01.gravity import ( gravity_potential_point, gravity_effect_point)

def gravity_potential_point(x, xm, m, G=6.674e-11):
    x = np.array(x, dtype=float)
    xm = np.array(xm, dtype=float)
    r = np.linalg.norm(x - xm)
    return G * m / r

def gravity_effect_point(x, xm, m, G=6.674e-11):
    x = np.array(x, dtype=float)
    xm = np.array(xm, dtype=float)
    r = np.linalg.norm(x - xm)
    return G * m * (x[2] - xm[2]) / r**3

def compute_fields(xg, yg, zp, xm, m):
    U = np.zeros((xg.shape[0], xg.shape[1], len(zp)))
    g = np.zeros_like(U)

    for k, z in enumerate(zp):
        for i in range(xg.shape[0]):
            for j in range(xg.shape[1]):
                x_obs = np.array([xg[i, j], yg[i, j], z])
                U[i, j, k] = gravity_potential_point(x_obs, xm, m)
                g[i, j, k] = gravity_effect_point(x_obs, xm, m)

    return U, g

def main():

    m = 1.0e7
    xm = np.array((0.0, 0.0, -10.0))
    zp = [0.0, 10.0, 100.0]

    x_25, y_25 = np.meshgrid(
        np.linspace(-100.0, 100.0, 9),
        np.linspace(-100.0, 100.0, 9),
    )
    U_25, g_25 = compute_fields(x_25, y_25, zp, xm, m)

    x_5, y_5 = np.meshgrid(
        np.linspace(-100.0, 100.0, 41),
        np.linspace(-100.0, 100.0, 41),
    )
    U_5, g_5 = compute_fields(x_5, y_5, zp, xm, m)

    # colour bar lims
    Umin = min(U_25.min(), U_5.min())
    Umax = max(U_25.max(), U_5.max())

    gmin = min(g_25.min(), g_5.min())
    gmax = max(g_25.max(), g_5.max())

    # plot func
    def plot_figure(xg, yg, U, g, title_prefix):

        fig, axes = plt.subplots(
            nrows=3, ncols=2, figsize=(10, 12), constrained_layout=True
        )

        for k, z in enumerate(zp):

            # --- Gravity potential (plasma) ---
            cs1 = axes[k, 0].contourf(
                xg, yg, U[:, :, k],
                levels=30,
                cmap="plasma",
                vmin=Umin, vmax=Umax
            )
            fig.colorbar(cs1, ax=axes[k, 0])
            axes[k, 0].scatter(
                xg, yg, s=1, c="k", alpha=0.3
            )
            axes[k, 0].set_title(f"Gravity potential (z = {z} m)")
            axes[k, 0].set_ylabel("y (m)")
            axes[k, 0].axis("equal")

            # --- Gravity effect (viridis) ---
            cs2 = axes[k, 1].contourf(
                xg, yg, g[:, :, k],
                levels=30,
                cmap="viridis",
                vmin=gmin, vmax=gmax
            )
            fig.colorbar(cs2, ax=axes[k, 1])
            axes[k, 1].scatter(
                xg, yg, s=1, c="k", alpha=0.3
            )
            axes[k, 1].set_title(f"Gravity effect $g_z$ (z = {z} m)")
            axes[k, 1].axis("equal")

        axes[-1, 0].set_xlabel("x (m)")
        axes[-1, 1].set_xlabel("x (m)")

        fig.suptitle(title_prefix, fontsize=14)
        plt.show()

   # plot
    plot_figure(x_25, y_25, U_25, g_25, "25 m Grid Spacing")
    plot_figure(x_5, y_5, U_5, g_5, "5 m Grid Spacing")

if __name__ == "__main__":
    main()
