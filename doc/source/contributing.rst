Contributing to VAST
====================

**VAST** development occurs `on GitHub <https://github.com/DESI-UR/VAST/>`_ 
using the standard GitHub workflow.  See the `GitHub documentation 
<https://docs.github.com/en>`_ if you are new to git or GitHub.



Feedback / Reporting Problems
-----------------------------

To give feedback, request new features, or report problems, please `open an 
issue on GitHub <https://github.com/DESI-UR/VAST/issues>`_.



Contributing Code or Documentation
----------------------------------

To contribute to **VAST**, clone the GitHub repository::

    git clone https://github.com/DESI-UR/VAST.git
    cd VAST/

After making changes, you can install the package with dependencies for the 
development version by running::

    pip install ".[dev]"
 
You can build the documentation by running::

    pip install ".[docs]"

If you plan on making an extensive set of changes, first `open an issue on 
GitHub <https://github.com/DESI-UR/VAST/issues>`_.  This will help coordinate 
your work with others.  It may turn out that your feature requests are already 
in progress.

Once you want to commit your changes, `submit a pull request 
<https://github.com/DESI-UR/VAST/pulls>`_.




Testing
-------

**VAST** uses the `pytest <https://docs.pytest.org>`_ package for automated
testing.  You may run the unit tests yourself manually by executing::

    pytest

at the command line in the root folder of the **VAST** source.
