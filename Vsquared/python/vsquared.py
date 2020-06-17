import sys
import getopt
from zobov import Zobov

args = sys.argv[1:]
opts, args = getopt.getopt(args,'c:m:wv',['config=','method=','save_intermediate','visualize'])

method = 0
save_intermediate = False
visualize = False

for o, a in opts:
    if o in ('-c','--config'):
        config_file = a
    if o in ('-m','--method'):
        method = a
    if o in ('-w','--save_intermediate'):
        save_intermediate = True
    if o in ('-v','--visualize'):
        visualize = True

newZobov = Zobov(config_file,0,3,save_intermediate=save_intermediate,visualize=visualize)

newZobov.sortVoids(method=method)

newZobov.saveVoids()

newZobov.sortVoids()

if visualize:
    newZobov.preViz()
