import os 
import numpy as np 
import matplotlib.pyplot as plt 
import sys, pathlib 
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scipy.io import ( savemat, loadmat, ) 
from src.goph547lab01.gravity import ( gravity_potential_point, gravity_effect_point)

m = 1.0e7 
G = 6.674e-11
zp = [0.0, 10.0, 100.0]

x_25, y_25 = np.meshgrid( np.linspace(-100.0, 100.0, 9), np.linspace(-100.0, 100.0, 9),) 

x_5, y_5 = np.meshgrid(np.linspace(-100.0, 100.0, 41), np.linspace(-100.0, 100.0, 41),)


def generate_mass_set():

    masses = np.zeros(5)
    xm = np.zeros((5,3))

    pos = np.array([0,0,-10])
    mass_position = np.zeros(3)

    #Compute first 4 masses and positions
    masses[:4] = np.random.normal(m/5, m/100, 4)
    xm[:4,0] = np.random.normal(0, 20, 4)
    xm[:4,1] = np.random.normal(0, 20, 4)
    xm[:4,2] = np.random.normal(-10,2,4)

    masses[4] = m - np.sum(masses[:3])
    mass_position[0] = np.sum(masses[:4] * xm[:4,0])
    mass_position[1] = np.sum(masses[:4] * xm[:4,1])
    mass_position[2] = np.sum(masses[:4] * xm[:4,2])

    xm[4] = (m * pos - mass_position) / masses[4]

    return masses, xm

mass_set_1 = generate_mass_set()[0]
mass_set_2 = generate_mass_set()[0]
mass_set_3 = generate_mass_set()[0]
xm = generate_mass_set()[1]

savemat('mass_set_1.mat', {'mass_set_1': mass_set_1})
savemat('mass_set_2.mat', {'mass_set_2': mass_set_2})
savemat('mass_set_3.mat', {'mass_set_3': mass_set_3})

def compute_fields(xg, yg, zp, xm, masses):
    U = np.zeros((xg.shape[0], xg.shape[1], len(zp)))
    g = np.zeros_like(U)

    for k, z in enumerate(zp):
        for i in range(xg.shape[0]):
            for j in range(xg.shape[1]):
                x_obs = np.array([xg[i, j], yg[i, j], z]) 
                for mi in range(len(masses)): 
                    mi_masses = masses[mi]
                    U[i, j, k] += gravity_potential_point(x_obs, xm[mi], mi_masses) 
                    g[i, j, k] += gravity_effect_point(x_obs, xm[mi], mi_masses)
                
    return U, g 

def plot_figure(xg, yg, U, g, title_prefix, Umin, Umax, gmin, gmax):

        fig, axes = plt.subplots(
            nrows=3, ncols=2, figsize=(10, 12), constrained_layout=True
        )

        for k, z in enumerate(zp):

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


def main(): 
    masses, xm = generate_mass_set() 

    U25, g25 = compute_fields(x_25,y_25, zp, masses, xm) 

    U5, g5 = compute_fields(x_5,y_5, zp, masses, xm) 

    Umin = min(U25.min(), U5.min())
    Umax = max(U25.max(), U5.max())
    gmin = min(g25.min(), g5.min())
    gmax = max(g25.max(), g5.max())

    plot_figure(x_25, y_25, U25, g25, "25 m Grid – Multi-Mass", Umin, Umax, gmin, gmax)
    plot_figure(x_5, y_5, U5, g5, "5 m Grid – Multi-Mass", Umin, Umax, gmin, gmax)

if __name__ == "__main__":
    main()