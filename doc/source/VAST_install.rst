.. role:: raw-html(raw)
    :format: html


.. _VAST-install:

Installation
============

Operating System Support & Requirements
---------------------------------------

Linux
^^^^^

**VoidFinder** is currently only fully supported on Linux, but may run on 
OSX and other Unix variants.

:raw-html:`<strong>V<sup>2</sup></strong>` is fully supported on Linux.

**VoidRender** is not expected to work well with the Wayland
graphical backend.  VoidRender works well on the older and more stable Xorg 
backend, on Intel, Nvidia, and AMD graphical drivers.  Note that for extreme
performance settings (very high numbers of voids or a ``SPHERE_TRIANGULARIZATION_DEPTH``
greater than 3), a discrete gpu from AMD or Nvidia is highly recommended for a smooth
visualization.

It may be helpful to install the following for **VoidRender** on Ubuntu::

    sudo apt-get install mesa-common-dev libgl1-mesa-dev libgl1-mesa-dev
    
Vispy can be especially tricky to get running, for testing in a python terminal you can::

    python
    >>>import vispy
    >>>vispy.test()
    
On Ubuntu 20.04 LTS it seems to work well with PyQT5::

    pip install PyQT5
    sudo apt install qt5-default


OSX
^^^

The authors have successfully run **VoidFinder** with full 
multi-processing power on Mac/OSX, but until Apple guarantees a POSIX- or Single 
Unix Specification-compliant ``fork()`` system call, full multi-processing 
**VoidFinder** cannot be guaranteed.  However, the single-process version of 
**VoidFinder** should always work on OSX to the best of our knowledge, and (as 
of March 2022), **VoidFinder** seems to be working correctly on OSX.

:raw-html:`<strong>V<sup>2</sup></strong>` runs on OSX.

**VoidRender** is expected to work on OSX using the Intel graphical drivers.


Windows
^^^^^^^

The authors have encountered difficulty compiling the **VAST** Cython code 
on the Windows platform, and more importantly the ``fork()`` system call is not 
supported by Windows.  Given its small popularity as a scientific computing 
platform, we have no plans (as of May 2022) to support Windows.  




Prerequisite Libraries
----------------------

**VoidFinder** uses Cython to speed up its operations, so when building from source
you will need a C compiler like gcc.

**VoidRender** requires OpenGL :math:`\geq` 1.2.




Building VAST
-------------

**VAST** does not yet have any pre-built wheels or distribution packages, 
so clone the repository from https://github.com/DESI-UR/VAST.

**VAST** will install like a normal python package via the shell command::

    pip install .
    
Older versions of python and VAST (prior to 2025 or so) used the `python setup.py install` method but the preferred python ecosystem method has changed to `pip install`
    

    
It is important to remember that this will attempt to install 
`vast` into the ``site-packages`` directory of whatever python 
environment that you are using.  To check on this, you can type::

    which python
    
into a normal unix shell and it will give you a path like ``/usr/bin/python`` or 
``/opt/anaconda3/bin/python``, which lets you know which python binary your 
``python`` command actually points to.


Developing VoidFinder
^^^^^^^^^^^^^^^^^^^^^

If you are actively developing **VAST**, you can install the package via::

    pip install -e .
    
This replaces the older `python setup.py develop` method.
    
which installs a symlink into your python environment's ``site-packages`` 
directory, and the symlink points back to wherever your local copy of the 
**VAST** directory is.

If you are developing **VAST** and need to rebuild the cython, from the 
``VAST`` directory run::

    python setup.py build_ext --inplace

In contrast to the newer `pip install` methods, this `python setup.py build_ext --inplace` method should still work fine for its purpose of just rebuilding the cython inplace.


Occasionally, it can be helpful to know the following command::

    cython -a *.pyx
    
which can be run from within the directory where the .pyx files live 
(currently ``VAST/python/vast/voidfinder/``) to "manually" build the cython 
(.pyx) files.


Installing VAST without admin privileges
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are working in an environment where you cannot install **VAST**, or 
you do not have permissions to install it into the python environment that you 
are using, add ``--user`` to your choice of build from above.  For example:: 

    pip install . --user

The old way was `python setup.py develop --user`
    
    
    
    
