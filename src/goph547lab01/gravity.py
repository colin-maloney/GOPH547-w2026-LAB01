import numpy as np 

def gravity_potential_point(x, xm, m, G=6.674e-11): 
    """ compute gravity potential due to a point masss 
    
    Parameters 
    x : array_like, shape(3, ) => coordinates of survey point
    xm : array_like, shape(3, ) => coordinates of mass anommaly
    m : float => mass of anommaly
    G : gravitational constant
    
    Retruns 
    gravity potential at x due to annomally at xm 
    """ 
    x = np.array(x, dtype=float) 
    xm = np.array(xm, dtype=float) 
    r = np.linalg.norm(x - xm) 
    return G * m/r 

def gravity_effect_point(x, xm, m , G=6.674e-11): 
    """ compute gravity effect due to a point masss 
    
    Parameters 
    x : array_like, shape(3, ) => coordinates of survey point
    xm : array_like, shape(3, ) => coordinates of mass anommaly
    m : float => mass of anommaly
    G : gravitational constant
    
    Retruns 
    gravity effect at x due to annomally at xm 
    """  
    x = np.array(x, dtype=float) 
    xm = np.array(xm, dtype=float) 
    r = np.linalg.norm(x - xm) 
    return G * m * (x[2]-xm[2])/ r**3 