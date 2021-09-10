'''
map_recontructed.bin ---> map_reconstructed.fits
Transform DR16 S82 reconstructed maps file inside 'map_reconstructed.bin' to a file format VoidFinder can work on.
File format: x,y,z,delta
RA in [-43,45]
DEC in  [-1.25,1.25]
Z in [2.1,3.2]

Read the column of delta values given inside the map_reconstructed.bin file and open it up as a data table of x,y,z.
Shift this whole box of delta values by the x,y,z coordinates of the point with  RA=-43°, DEC=-1.25° and z=2.1

'''

### Importing the necessary packages
import os
import numpy as np
from astropy.table import Table
import scipy.integrate
from astropy import constants as const
from scipy.integrate import quad as quad
import time

### In and out directories
in_directory = '/scratch/sbenzvi_lab/boss/dr16/reconstructed_maps/'
out_directory = '/scratch/ierez/IGMCosmo/VoidFinder/data/DR16S82_H/reconstructed/'
os.chdir(in_directory)

### Input file name
input_filename='map_reconstructed.bin'

### Reading the input file
def read_map_file(map_name,shape_map):
    with open(map_name,'r') as f:
        map_data = np.fromfile(f,dtype=np.float64).reshape(shape_map)
    return(map_data)

map_name = 'map_reconstructed.bin'
shape_map = (6360, 180, 834)   #because higher in resolution

map_3d = read_map_file(map_name,shape_map) #in 3D data format

print('Read the file')

### Calculate the shift corresponding to RA=-43°, DEC=-1.25° and z=2.1

ra_ref = -43 #degrees
dec_ref = -1.25 #degrees
redshift_ref = 2.1

Omega_M = 0.3147
c = const.c.to('km/s').value

def Distance(z,omega_m = Omega_M,h = 1):
    dist = np.ones(len(z))
    H0 = 100*h
    for i,redshift in enumerate(z):
        a_start = 1/(1+redshift)
        I = quad(f,a_start,1,args=omega_m)
        dist[i] = I[0]*(c/H0)
    return dist

def f(a,omega_m = Omega_M):
     return 1/(np.sqrt(a*omega_m*(1+((1-omega_m)*a**3/omega_m)))) 

comoving = Distance([redshift_ref],Omega_M,1)

DtoR = np.pi/180.
ra_radian = ra_ref*DtoR
dec_radian = dec_ref*DtoR

x_ref = comoving*np.cos(ra_radian)*np.cos(dec_radian)
y_ref = comoving*np.sin(ra_radian)*np.cos(dec_radian)
z_ref = comoving*np.sin(dec_radian)

### Collecting that data in a new Table while shifting from (0,0,0) to (x_ref,y_ref,z_ref)

def get_x(i):
    return i+x_ref

def get_y(j):
    return j+y_ref

def get_z(k):
    return k+z_ref


print('Alive before for loop')

main = Table(names=('x','y','z','delta'))

for i in range(6360):
    for j in range(180):
        for k in range(834):
            main.add_row((get_x(i),get_y(j),get_z(k),map_3d[i,j,k]))


print('Alive after for loop')

### Write the output file 


'''                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                              
map_reconstructed.dat ---> map_reconstructed.fits                                                                                                                                                                                             
                                                                                                                                                                                                                                              
Read map_reconstructed.dat data                                                                                                                                                                                                               
Remove the delta readings equal to 0 because they were set to zero                                                                                                                                                                            
by the tomographic map reconstruction algorithm.                                                                                                                                                                                              
Randomize the row order to make the separation calculations faster.                                                                                                                                                                           
                                                                                                                                                                                                                                              
'''

np.random.seed(15)

main=main[main['delta']!=0]

print(len(main))

print('Removed 0s.')

np.random.shuffle(main)

print('Randomized the row order.')

main.write('data_reconstructed.fits', format='fits', overwrite=True)   

print('Generated the map_reconstructed.fits file.')
