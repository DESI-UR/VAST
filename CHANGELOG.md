# Change Log
Log of changes for VAST versions.

=======
### 1.7.6
- Option for fast, maximal-spheres-only version of VoidFinder suitable for forward modeling with large simulation suites

### 1.7.5
- Expanded options for specifying the galaxy input to V2 and for plotting V2 voids with the slice plot code

### 1.7.4
- Updated galaxy_membership method in the void catalog classes to optionally return the indices of wall galaxies

### 1.7.3
- Updated VoidFinder to allow option for not saving galaxies to void-finding output

### 1.7.2
- Updated void catalog class to let users search for the void membership of custom coordinates,
  along with small bug fixes

### 1.7.1

- Bugfix: updated pyx files to account for changes in Cython math functions ([PR #127](https://github.com/DESI-UR/VAST/pull/127)).

### 1.7.0
- Feature: Added parallel computing features for V2. Added non-periodic cubic mode and edge-void calculation for V2

### 1.6.2
- Updated V2 unit tests to work on Python 3.8, 3.9, 3.10, 3.11 with minor
  change from np.isclose to checking for 1% differences between results

### 1.6.1
- Updated VF unit tests to check maximal positions and radii and hole overlap
  against designated maximals instead of utilizing the more brittle file
  based output of previous VF runs

## 1.6.0
- Feature: Added void catalog class for reading void-finding fits output and performing analysis.

## 1.5.0
- Feature: Replaced the multiple txt output files for VoidFinder and V2 with a single fits file.

## 1.4.3
- VF Bugfix:  Found an issue finding 3rd bounding galaxy for a hole when galaxies are colinear with the 
              hole center as in some synthetic datasets.
              Changed Period Mode strategy to add an offset to existing galaxies instead of allocating
              additional memory as new cells.  May want to revisit strategy, but appears to be working well,
              in non-periodic mode this offset should always be 0.0.  VF on SDSS DR7 finds 900 voids now using the 
              example script.

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
