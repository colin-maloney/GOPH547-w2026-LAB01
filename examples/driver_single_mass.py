import numpy as np 
import matplotlib as plt 

from goph547lab01.gravity import ( gravity_potential_point, gravity_effect_point) 

def main(): 

    m = 1.0e7 
    xm = np.array((0.0, 0.0, -10.0)) 

    x_25, y_25 = np.meshgrid( np.linspace(-100.0, 100.0, 41), np.linspace(-100, 100, 9)) 

    x_6, y_5 = np.meshgrid( (np.linspace(-100.0, 100.0, 41), np.linspace(-100.0, 100.0, 41))) 

    zp = [0.0, 10.0, 100.0] 

    U_25 = np.zeros((x_25.shape[0], x_25.shape[1],len(zp) )) 
    g_25 = np.zeros((x_25.shape[0], x_25.shape[1], len(zp))) 
    xs = x_25[0, :] 
    ys = y_25[:, 0]