import numpy as np 
import matplotlib.pyplot as plt 

from goph547lab01.gravity import ( gravity_potential_point, gravity_effect_point) 

def main(): 

    m = 1.0e7 
    xm = np.array((0.0, 0.0, -10.0)) 

    x_25, y_25 = np.meshgrid( np.linspace(-100.0, 100.0, 41), 
                            np.linspace(-100, 100, 9)) 

    x_5, y_5 = np.meshgrid(np.linspace(-100.0, 100.0, 41), 
                            np.linspace(-100.0, 100.0, 41)) 

    zp = [0.0, 10.0, 100.0] 

    U_25 = np.zeros((x_25.shape[0], x_25.shape[1],len(zp) )) 
    g_25 = np.zeros((x_25.shape[0], x_25.shape[1], len(zp))) 
    xs = x_25[0, :] 
    ys = y_25[:, 0] 

for k, z in enumerate(zp):
    for i in range(x_25.shape[0]):
        for j in range(x_25.shape[1]):
            x_obs = np.array([x_25[i, j], y_25[i, j], z])

            U_25[i, j, k] = gravity_potential_point(x_obs, xm, m)
            g_25[i, j, k] = gravity_effect_point(x_obs, xm, m) 

k = 0  # index for z = 0

plt.figure(figsize=(6, 5))
cs = plt.contourf(x_25, y_25, U_25[:, :, k], levels=30, cmap="viridis")
plt.colorbar(cs, label="Gravity potential (m²/s²)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title(f"Gravity potential at z = {zp[k]} m")
plt.axis("equal")
plt.show()


plt.figure(figsize=(6, 5))
cs = plt.contourf(x_25, y_25, g_25[:, :, k], levels=30, cmap="seismic")
plt.colorbar(cs, label="Gravity effect $g_z$ (m/s²)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title(f"Gravity effect at z = {zp[k]} m")
plt.axis("equal")
plt.show()
