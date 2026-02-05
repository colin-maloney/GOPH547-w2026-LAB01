import os 
import numpy as np 
import matplotlib.pyplot as plt 
import sys, pathlib 
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scipy.io import ( savemat, loadmat, ) 
from src.goph547lab01.gravity import ( gravity_potential_point, gravity_effect_point)

data = loadmat('anomaly_data.mat') 

x = data['x'] 
y = data['y'] 
z = data['z'] 
rho = data['rho'] 

cell_vol = 2 **3 

mass_cell = cell_vol * rho 
net_mass = np.sum(mass_cell) 

xa = np.sum(mass_cell * x) / net_mass 
ya = np.sum(mass_cell * y) / net_mass 
za = np.sum(mass_cell * z) / net_mass 

anomaly_pos = [xa, ya, za] 

x_vec = x[0, :, 0] 
y_vec = y[:, 0, 0] 
z_vec = z[0, 0, :] 

rho_xz = np.mean(rho, axis=0) 
X_xz, Z_xz = np.meshgrid(x_vec, z_vec) 

rho_yz = np.mean(rho, axis=1) 
Y_yz, Z_yz = np.meshgrid(y_vec, z_vec) 

rho_xy = np.mean(rho, axis=2) 
X_xy, Y_xy = np.meshgrid(x_vec, y_vec)  

fig, axes = plt.subplots(3, 1, figsize=(8, 12))

c0 = axes[0].contourf(X_xz, Z_xz, rho_xz.T, levels=20)
axes[0].plot(xa, za, 'xk', markersize=3)
axes[0].set_xlabel('x (m)')
axes[0].set_ylabel('z (m)')
axes[0].set_xlim(-20,20)
axes[0].set_ylim(0,-20)
axes[0].set_title('Anomaly in xz-plane')
cbar0 = fig.colorbar(c0, ax=axes[0])

c1 = axes[1].contourf(Y_yz, Z_yz, rho_yz.T, levels=20)
axes[1].plot(ya, za, 'xk', markersize=3)
axes[1].set_xlabel('y (m)')
axes[1].set_ylabel('z (m)')
axes[1].set_xlim(-20,20)
axes[1].set_ylim(0,-20)
axes[1].set_title('Anomaly in yz-plane')
fig.colorbar(c1, ax=axes[1])

c2 = axes[2].contourf(X_xy, Y_xy, rho_xy.T, levels=20)
axes[2].plot(xa, ya, 'xk', markersize=3)
axes[2].set_xlabel('x (m)')
axes[2].set_ylabel('y (m)')
axes[2].set_xlim(-20,20)
axes[2].set_ylim(25,-25)
axes[2].set_title('Anomaly in xy-plane')
cbar2 = fig.colorbar(c2, ax=axes[2]) 

X_xz_min, X_xz_max = -20, 20 
Z_xz_min, Z_xz_max = -20, 0 

Y_yz_min, Y_yz_max = X_xz_min, X_xz_max
Z_yz_min, Z_yz_max = Z_xz_min, Z_xz_max

X_xy_min, X_xy_max = X_xz_min, X_xz_max
Y_xy_min, Y_xy_max = -25,25 

ix_xz = np.where((x_vec >= X_xz_min) & (x_vec <= X_xz_max))[0]
iz_xz = np.where((z_vec >= Z_xz_min) & (z_vec <= Z_xz_max))[0]

iy_yz = np.where((y_vec >= Y_yz_min) & (y_vec <= Y_yz_max))[0]
iz_yz = np.where((z_vec >= Z_yz_min) & (z_vec <= Z_yz_max))[0]

ix_xy = np.where((x_vec >= X_xy_min) & (x_vec <= X_xy_max))[0]
iy_xy = np.where((y_vec >= Y_xy_min) & (y_vec <= Y_xy_max))[0]

rho_crop_xz = rho_xz[np.ix_(ix_xz, iz_xz)]
rho_crop_yz = rho_yz[np.ix_(iy_yz, iz_yz)]
rho_crop_xy = rho_xy[np.ix_(ix_xy, iy_xy)]

mean_rho_xz = np.mean(rho_crop_xz)
mean_rho_yz = np.mean(rho_crop_yz)
mean_rho_xy = np.mean(rho_crop_xy) 

zp = [0,100] 
m = net_mass 

x_5, y_5 = np.meshgrid(np.linspace(-100,100,41), np.linspace(-100, 100, 41)) 

gz = np.zeros((x_5.shape[0], x_5.shape[1], len(zp)))

for k, z_obs in enumerate(zp):
    for i in range(x_5.shape[0]):
        for j in range(x_5.shape[1]):
            x = np.array([x_5[i, j], y_5[i, j], z_obs])
            gz[i, j, k] = gravity_effect_point(x, anomaly_pos, m).item()

gz_min = np.min(gz)
gz_max = np.max(gz)

fig, axes = plt.subplots(2, 1, figsize=(8, 12))

for k, z_obs in enumerate(zp):
    ax = axes[k]
    c = ax.contourf(x_5, y_5, gz[:, :, k], levels=20, vmin=gz_min, vmax=gz_max, cmap = 'viridis_r')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'Gravitational Effect at z = {z_obs} m')
    fig.colorbar(c, ax=ax)

fig.suptitle('Anomaly Gravity Effect at \nGround and Airborne Observation', fontsize=16)
plt.savefig('Anomaly Gravity Effect Forward Modelling.png')

zp_all = [0.0, 1.0, 100.0, 110.0]

gz = np.zeros((x_5.shape[0], x_5.shape[1], len(zp_all)))

for k, z_obs in enumerate(zp_all):
    for i in range(x_5.shape[0]):
        for j in range(x_5.shape[1]):
            x_obs = np.array([x_5[i, j], y_5[i, j], z_obs])
            g_vec = gravity_effect_point(x_obs, anomaly_pos, m)
            gz[i, j, k] = gravity_effect_point(x_obs, anomaly_pos, m)  

z0, z1, z100, z110 = 0, 1, 2, 3

dgz_dz_0 = (gz[:, :, z1] - gz[:, :, z0]) / 1.0
dgz_dz_100 = (gz[:, :, z110] - gz[:, :, z100]) / 10.0

fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

gz_min = gz.min()
gz_max = gz.max()

for k, z_obs in enumerate(zp_all):
    ax = axes.flat[k]

    c = ax.contourf(
        x_5, y_5, gz[:, :, k],
        levels=20,
        vmin=gz_min, vmax=gz_max,
        cmap='viridis_r'
    )

    ax.set_title(f'$g_z$ at z = {z_obs:.0f} m')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')

    fig.colorbar(c, ax=ax)

fig.suptitle(
    'Vertical Gravity Effect $g_z$ at Four Survey Elevations',
    fontsize=16
)

plt.show() 

def plot_fd_first_derivative(
    xg, yg, dgz_dz,
    z_level,
    vmin=None, vmax=None,
    cmap='seismic'
):
    """
    Plot first-order finite-difference estimate of ∂gz/∂z.

    Parameters
    ----------
    xg, yg : 2D arrays
        Grid coordinates.
    dgz_dz : 2D array
        First-order FD vertical derivative of gz.
    z_level : float
        Elevation at which the derivative is evaluated (m).
    vmin, vmax : float, optional
        Color scale limits.
    cmap : str
        Matplotlib colormap.
    """

    fig, ax = plt.subplots(figsize=(6, 5))

    c = ax.contourf(
        xg, yg, dgz_dz,
        levels=20,
        vmin=vmin, vmax=vmax,
        cmap=cmap
    )

    ax.set_title(rf'First-Order FD $\partial g_z / \partial z$ at z = {z_level} m')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')

    fig.colorbar(c, ax=ax, label=r'$\partial g_z / \partial z$ (s$^{-2}$)')
    plt.show()

plot_fd_first_derivative(
    x_5, y_5,
    dgz_dz_0,
    z_level=0
)

plot_fd_first_derivative(
    x_5, y_5,
    dgz_dz_100,
    z_level=100
)


d2gz_dz2_0_fd = (gz[:, :, z1] - gz[:, :, z0]) / (1.0**2)
d2gz_dz2_100_fd = (gz[:, :, z110] - gz[:, :, z100]) / (10.0**2)

dx = x_5[0,1] - x_5[0,0]
dy = y_5[1,0] - y_5[0,0] 

def laplace_vertical(gz_slice, dx, dy):
    d2gz_dx2 = np.gradient(np.gradient(gz_slice, dx, axis=1), dx, axis=1)
    d2gz_dy2 = np.gradient(np.gradient(gz_slice, dy, axis=0), dy, axis=0)
    return -(d2gz_dx2 + d2gz_dy2)

d2gz_dz2_0_lap = laplace_vertical(gz[:, :, z0], dx, dy)
d2gz_dz2_100_lap = laplace_vertical(gz[:, :, z100], dx, dy)

fig, axes = plt.subplots(2, 1, figsize=(8, 12), constrained_layout=True)

for ax, data, z in zip(
    axes,
    [d2gz_dz2_0_lap, d2gz_dz2_100_lap],
    [0, 100]
):
    vmax = np.max(np.abs(data))
    vmin = -vmax

    c = ax.contourf(
        x_5, y_5, data,
        levels=20,
        vmin=vmin, vmax=vmax,
        cmap='seismic'
    )
    ax.set_title(rf'$\partial^2 g_z / \partial z^2$ at z = {z} m')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    fig.colorbar(c, ax=ax)

fig.suptitle(
    r'Second Vertical Derivative from Laplace Equation',
    fontsize=16
)

plt.show()

