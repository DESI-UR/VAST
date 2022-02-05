"""Some code for interacting with git.
"""

import re
from os.path import abspath, exists, isdir, isfile, join

def get_version():
    """Get the value of ``__version__`` without having to import the package.

    Returns
    -------
    :class:`str`
        The value of ``__version__``.
    """
    ver = 'unknown'
    try:
        version_dir = find_version_directory()
    except IOError:
        return ver

    version_file = join(version_dir, '_version.py')
    with open(version_file, 'r') as f:
        for line in f.readlines():
            mo = re.match("__version__ = '(.*)'", line)
            if mo:
                ver = mo.group(1)
    return ver

def find_version_directory():
    """Return the name of a directory containing version information.
    Looks for files in the following places:
    * python/vast/_version.py
    * vast/_version.py

    Returns
    -------
    :class:`str`
        Name of a directory that can or does contain version information.

    Raises
    ------
    IOError
        If no valid directory can be found.
    """
    packagename = 'vast'
    setup_dir = abspath('.')

    if isdir(join(setup_dir, 'python', packagename)):
        version_dir = join(setup_dir, 'python', packagename)
    elif isdir(join(setup_dir, packagename)):
        version_dir = join(setup_dir, packagename)
    else:
        raise IOError('Could not find a directory containing version information!')
    return version_dir

