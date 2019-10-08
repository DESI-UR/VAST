#absolute magnitude & distance functions

import numpy as np
from scipy.integrate import quad



c = 299792




def f(a,omega_m):
     return 1/(np.sqrt(a*omega_m*(1+((1-omega_m)*a**3/omega_m))))



def Distance(z,omega_m,h):
    dist = np.ones(len(z))
    H0 = 100*h
    for i,redshift in enumerate(z):
        a_start = 1/(1+redshift)
        I = quad(f,a_start,1,args=omega_m)
        dist[i] = I[0]*(c/H0)
    return dist



def Rabsmag(omega_m,app_mag,z,h):
    comovingdist = Distance(z,omega_m,h)*1e6
    return app_mag -(5*np.log10(comovingdist))+5


