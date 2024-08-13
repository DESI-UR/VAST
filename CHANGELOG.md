# Change Log
Log of changes for VAST versions.



## 1.4.3
- VF Bugfix:  Found an issue finding 3rd bounding galaxy for a hole when galaxies are colinear with the 
              hole center as in some synthetic datasets.
              Changed Period Mode strategy to add an offset to existing galaxies instead of allocating
              additional memory as new cells.  May want to revisit strategy, but appears to be working well,
              in non-periodic mode this offset should always be 0.0.

## 1.4.2
- VF Bugfix:  Galaxies exactly on the "far" edge of the survey (cells furthest from grid_origin) causing problems
              with Cell ID Generation - resulting in a cell which "should" be empty but the hole_cell_ID_dict cant
              filter them because their Cell ID is promoted by 1, even though we know they belong to the original
              survey even in periodic mode - fixed by "demoting" any galaxy exactly on the far edge of the survey 
              to its adjacent interior cell
           
- VF Bugfix:  Multiple places where the cpython 'mmap' object was raising goofy BufferErrors which were not actually
              a problem - this is more of a cpython implementation detail problem, added a workaround by deleting
              the python mmap and re-opening the python mmap object, probably wont incur any noticable performance loss
              Comments added in the code where these problems exist and have been worked around
           
- VF Bugfix:  Cell ID generation was stopping after any batch which had 0 valid hole start locations instead of 
              continuing until the entire survey was run through.  Hadn't noticed this problem earlier because at
              large batch sizes (aka default 10,000) it is generally unlikely to return a full 0 valid start hole 
              locations


## 1.4.1
- Did Stuff

## 1.3.1 (unreleased)
- Added duplicate galaxy check in VF

## 1.3.0 (2023-01-05)
- Fixed bug in V2 that allowed zero-volume cells in zones, added 3rd ZOBOV pruning method

## 1.2.2 (2022-10-11)
- Fixed bug in VF regarding incorrect handling of co-linear and co-planar neighbor galaxies

## 1.2.0 (2022-09-16)
- JOSS publication version; begin tracking with changelog
