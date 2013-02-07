#!/usr/bin/env python3
#encoding: utf-8

import sys
import getopt
import formula
import utils
import time
import pickle



help_message = '''
Input is taken from model.py and formula.py
'''


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def main(argv=None):
    
    forestOnly = False
    strategyOnly = False
    
    # argument processing
    # TODO UI
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "fh:s", ["help"])
        except getopt.error as msg:
            raise Usage(msg)
        
        # option processing
        for option, value in opts:
            if option in ("-h", "--help"):
                raise Usage(help_message)
            if option == "-f":
                forestOnly = True
            if option == "-s":
                strategyOnly = True
                
    except Usage as err:
        print(sys.argv[0].split("/")[-1] + ": " + str(err.msg), file=sys.stderr)
        print("\t for help use --help", file=sys.stderr)
        return 2
        
    # strategy syntesis
    start = time.clock();
    
    if not strategyOnly:
        # construct the forest
        phi = formula.phi()
        if forestOnly:
            # serialize the forest for the further use
            phi.export(utils.outputPath() + "phi.graphml")
            with open(utils.outputPath() + "pickledTree.bin", 'wb') as f:
                f.truncate(0)
                pickle.Pickler(f, pickle.HIGHEST_PROTOCOL).dump(phi)
            with open(utils.outputPath() + "pickledUsedPerceptions.bin", 'wb') as f:
                f.truncate(0)
                pickle.Pickler(f, pickle.HIGHEST_PROTOCOL).dump(utils.usedPerceptions)
            with open(utils.outputPath() + "pickledUsedStates.bin", 'wb') as f:
                f.truncate(0)
                pickle.Pickler(f, pickle.HIGHEST_PROTOCOL).dump(utils.usedStates)
            return
    
    else:
        # only construct the strategy
        # load the serialized forest phi
        with open(utils.outputPath() + "pickledTree.bin", 'rb') as f:
            phi = pickle.Unpickler(f).load()
        with open(utils.outputPath() + "pickledUsedPerceptions.bin", 'rb') as f:  
            utils.usedPerceptions = pickle.Unpickler(f).load()
        with open(utils.outputPath() + "pickledUsedStates.bin", 'rb') as f:   
            utils.usedStates = pickle.Unpickler(f).load()
    
    strategy = phi.totalStrategy()
    
    if strategy:
        strategy.export(utils.outputPath() + utils.strategyPath)
        print("Strategy exported")
    else:
        with open(utils.outputPath() + utils.strategyPath, 'w') as f:
            f.truncate(0)
        print("No solution")
    
    end = time.clock()
    print("time elapsed:", end - start)
    return

if __name__ == "__main__":
    sys.exit(main())

