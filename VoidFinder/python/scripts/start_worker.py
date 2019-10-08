









if __name__ == "__main__":
    
    import sys
    
    sys.path.insert(0, '/home/moose/VoidFinder/python/')
    
    CONFIG_PATH = "/tmp/voidfinder_config.pickle"
    
    from voidfinder._voidfinder import _main_hole_finder_worker
    
    _main_hole_finder_worker(CONFIG_PATH)