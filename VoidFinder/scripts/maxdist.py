###### finding max r value for dc 17

from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.io import fits
from vast.voidfinder.distance import z_to_comoving_dist

'''
z = [.7]
omega_m = 0.3 
h = 0.7345

dist = z_to_comoving_dist(z,omega_m,h)

print(dist)
'''


fig = plt.figure()

gal_file = fits.open('DESI_void_mock_2.fits') 
gal_data = Table(gal_file[1].data)

abs_mag = gal_data['rabsmag']
redshift = gal_data['redshift']

ax = fig.add_subplot(111)
ax.invert_yaxis()
ax.scatter(redshift,abs_mag,s=5,c='seagreen') 
ax.axhline(y=-20)

ax.set_title('DC17 - Absolute Magnitude vs. Redshift')
ax.set_xlabel('Z')
ax.set_ylabel('R Absolute Magnitude')
plt.show()


