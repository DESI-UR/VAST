#!/usr/bin/env python
"""
Driver program for Voronoi Voids (V^2) void finder module using the ZOBOV
algorithm.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from vast.vsquared.zobov import Zobov

p = ArgumentParser(description='Voronoi Voids (V^2) void finder.',
                   formatter_class=ArgumentDefaultsHelpFormatter)

p.add_argument('-m', '--method', default='0',
               help='Void-finding method (0,1,2,...) or (VIDE,ZOBOV,ZOBOV2,...) or (vide,zobov,zobov2,...)')
p.add_argument('-v', '--visualize', action='store_true', default=False,
               help='Enable void visualization.')
p.add_argument('-w', '--save_intermediate', action='store_true', default=False,
               help='Save intermediate files in void calculation.')
p.add_argument('-p', '--periodic', action='store_true', default=False,
               help='Use periodic boundary contitions.')

req = p.add_argument_group('required named arguments')
req.add_argument('-c', '--config', dest='config_file', required=True, default="DR7_config.ini",
                 help='V^2 config file (INI format).')

args = p.parse_args()

newZobov = Zobov(args.config_file, 0, 3,
                 save_intermediate=args.save_intermediate,
                 visualize=args.visualize,
                 periodic=args.periodic)

newZobov.sortVoids(method=args.method)

newZobov.saveVoids()

newZobov.saveZones()

if args.visualize:
    newZobov.preViz()
