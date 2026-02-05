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
fig.colorbar(c0, ax=axes[1])

c2 = axes[2].contourf(X_xy, Y_xy, rho_xy.T, levels=20)
axes[2].plot(xa, ya, 'xk', markersize=3)
axes[2].set_xlabel('x (m)')
axes[2].set_ylabel('y (m)')
axes[2].set_xlim(-20,20)
axes[2].set_ylim(25,-25)
axes[2].set_title('Anomaly in xy-plane')
cbar2 = fig.colorbar(c0, ax=axes[2]) 

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

