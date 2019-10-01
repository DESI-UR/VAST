
import sys
import os


if __name__ == "__main__":
    """
    For running in multi-process mode, the run_multi_process function above now uses
    the helper function _main_hole_finder_startup() and os.execv(), which in turn
    invokes this file _voidfinder.py as a script, and will enter this
    if __name__ == "__main__" block, where the real worker function is called.
    
    The worker then will load a pickled config file based on the 
    `config_path` argument which the worker can then use to
    configure itself and connect via socket to the main process.
    """
    
    voidfinder_dir = sys.argv[1]
    
    working_dir = sys.argv[2]
    
    worker_idx = sys.argv[3]
    
    config_path = sys.argv[4]
    
    #When this module was called with os.execv(), it was given the path to this file,
    #and when you do that I think python automatically adds the path to the sys.path
    #in position 0 so the current directory is part of the path.  HOWEVER, we want
    #the parent directory of this file to be part of the path, not this file itself.
    
    if voidfinder_dir in sys.path[0] and sys.path[0] != voidfinder_dir:
        
        sys.path[0] = voidfinder_dir
        
    if not (voidfinder_dir in sys.path):
        
        sys.path.insert(0, voidfinder_dir)
        
    os.chdir(working_dir)
    
    #print("WORKER STARTED WITH ARGS: ", sys.argv)
    #print(sys.path[0:10])
    
    from voidfinder._voidfinder import _main_hole_finder_worker
    
    _main_hole_finder_worker(worker_idx, config_path)
    
    
    
    
    
    
    
    
    