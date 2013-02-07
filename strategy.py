#!/usr/bin/env python3
#encoding: utf-8

class Strategy:
    """partial strategy: rules + domain"""
    def __init__(self):
        """dictionary of rules: perceptions as keys, actions as values"""
        self.rules = {}
        """ set of system states at which strategy works"""
        self.domain = set([])
    
    def export(self, fname):
        """Export the domain (list of states) and the rules into 'fname' file.
        """
        with open(fname, 'w') as f:
            f.truncate(0)
            sDomain = list(self.domain)
            sDomain.sort()
            f.write("Domain size:{0}\n".format(len(sDomain)))
            f.write("Domain:\n")
            for state in sDomain:
                f.write("{0}\n".format(state))
            f.write("\nRules:{0}\n".format(len(self.rules)))
            sPerceptions = list(self.rules.keys())
            sPerceptions.sort()
            for perc in sPerceptions:
                f.write("{0} -> {1}\n".format(perc, self.rules[perc]))
        return
    
