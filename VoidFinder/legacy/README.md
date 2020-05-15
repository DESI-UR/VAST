# Legacy VoidFinder implementation(s)

In this directory lives the old version of VoidFinder written in Fortran, as well as an old
python version.

## Fortran

To run the fortran, you'll need a compiler.  It compiles and runs successfully on gfortran from
gcc version 7.5.0 (Ubuntu 7.5.0-3ubuntu1~18.04)

To compile (see here for reference: https://gcc.gnu.org/wiki/GFortranGettingStarted):
`
gfortran -c dist.f indexx.f ran3.f sort_4.f
gofrtran voids_sdss_dr7.f dist.o indexx.o ran3.o sort_4.o -o voidfinder_fortran
`

Then to run, you'll need to create a directory in the same directory as `voidfinder_fortran`
titled `SDSSdr7` and inside that directory put the files `vollim_dr7_cbp_102709.dat` and
`cbpdr7mask.dat`

With the directory structure complete, run the fortran version of voidfinder like any other
binary file, cd into the directory where `voidfinder_fortran` lives and do `./voidfinder_fortran`


## Python

Also in this directory is the void_sdss.py file, the first python re-write of VoidFinder.  It is
only included for reference, the existing implementation is in the `VoidFinder/python` directory
and is many orders of magnitude faster and better documented.