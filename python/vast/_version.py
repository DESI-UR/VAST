
"""

Hopefully this makes sense to everyone - trying to stick with the fairly common
'major.minor.revision' GNU version number format here.  As a rule of thumb,
major is incremented when we make any huge updates (maybe adding a whole new
algorithm, or a complete re-write of an existing one), the minor
is incremented when we make any significant or backwards-incompatible changes,
and the 3rd number is for more general bug fixes, enhancements, quality
changes and so forth.

Note that this scheme is still pretty loosey-goosey and ultimately it still falls
on the package maintainers to debate and find consensus on each issue individually
but this can provide a useful frame of reference for the versioning.

Also when we hit something like 1.9.x we dont have to go to 2.0.0, we can go to 1.10.x
just in case anyone wondered about that like I used to.
"""

__version__ = '1.2.1'
