import numpy as np 
from goph547lab01.gravity import ( gravity_potential_point, gravity_effect_point) 

""" Test Gravity effect functions with arbitray values""" 
x = [1, 2, 3] 
xm = [4, 5, 6] 
m = 10 

# test gravity potential function 
U = gravity_potential_point(x, xm, m, G=6.674e-11) 
print(f"The gravity potneial felt at x by mass anommaly m is {U}") 

# test gravity effect funtion 
g = gravity_effect_point(x, xm, m, G=6.674e-11) 
print(f"the gravity effect felt at x by the mass annomaly m is{g}")