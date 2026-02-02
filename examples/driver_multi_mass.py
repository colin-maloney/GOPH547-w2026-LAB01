import os 
import numpy as np 
import matplotlib.pyplot as plt 
import sys, pathlib 
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scipy.io import ( savemat, loadmat, ) 
from src.goph547lab01.gravity import ( gravity_potential_point, gravity_effect_point)


def generate_mass_annomaly_sets(
    m_total=1.0e7,
    n_sets=3,
    outdir="examples"
):
    """
    Generate mass anomaly sets satisfying:
      sum(m_i) = m_total
      sum(m_i * x_i) = 0

    Saves each set as examples/mass_set_k/masses.mat
    """

    rng = np.random.default_rng()

    mu_m = m_total / 5.0
    sigma_m = m_total / 100.0

    mu_x = 0.0
    mu_y = 0.0
    mu_z = -10.0

    sigma_x = 20.0
    sigma_y = 20.0
    sigma_z = 2.0

    for k in range(n_sets):

        set_dir = os.path.join(outdir, f"mass_set_{k}")
        os.makedirs(set_dir, exist_ok=True)

        m4 = rng.normal(mu_m, sigma_m, size=4)

        x4 = rng.normal(mu_x, sigma_x, size=4)
        y4 = rng.normal(mu_y, sigma_y, size=4)
        z4 = rng.normal(mu_z, sigma_z, size=4)

        xm4 = np.column_stack((x4, y4, z4))

        m5 = m_total - np.sum(m4)

        moment = np.sum(m4[:, None] * xm4, axis=0)
        x5 = -moment / m5

        m_all = np.hstack((m4, m5))
        xm_all = np.vstack((xm4, x5))

        savemat(
            os.path.join(set_dir, "masses.mat"),
            {
                "m": m_all,
                "xm": xm_all,
            }
        )

        print(f"Generated mass_set_{k}")
        print(f"  Total mass: {np.sum(m_all):.3e}")
        print(f"  Centroid: {np.sum(m_all[:,None]*xm_all, axis=0)}")

def compute_fields(xg, yg, zp, xm, m):
    U = np.zeros((xg.shape[0], xg.shape[1], len(zp)))
    g = np.zeros_like(U)

def main(): 
    if ( not os.path.exists("examples/mass_set_0")
        or not os.path.exists("examples/mass_set_1") 
        or not os.path.exists("examples/mass_set_2")): 

        generate_mass_annomaly_sets() 

    x_25, y_25 = np.meshgrid(
        np.linspace(-100.0, 100.0, 9),
        np.linspace(-100.0, 100.0, 9),
    ) 
    x_5, y_5 = np.meshgrid(
        np.linspace(-100.0, 100.0, 41),
        np.linspace(-100.0, 100.0, 41),
    ) 

    zp = [0.0, 10.0, 100.0] 

    # load data set