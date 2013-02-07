#!/usr/bin/env python3
#encoding: utf-8
"""
model of common concesus in the ring of |agt|
"""

agt = 6
act = 2
st = 2
k = 2
mask = 7


def neig(a, n):
    """
    topology encoding => model-specific
    returns n-th neighbor of agent a
    """
    if n == 0:
        return (a - 1) % agt
    elif n == 1:
        return (a + 1) % agt
    else:
        return -1


def neig_1(a, na):
    """
    model-specific (faster)
    returns the index of a's neighbor na
    """
    if na == ((a - 1) % agt):
        return 0
    elif na == ((a + 1) % agt):
        return 1
    else:
        return -1   


def manif(q):
    """
    general
    returns the manifestation of individual state q 
    """
    return q & mask


def iState(prc):
    """"
    general
    returns the individual state from a perception
    """ 
    return prc[0]


def tran(prc, action):
    """
    transition function => model specific
    returns outcome of action act at perception prc
    """
    if action == 0:
        return iState(prc)
    else:
        return (iState(prc) + 1) % st           

def tran_1(l_state):
    """
    inverted tr.fun. => model specific
    returns list of rules with outcome l_state  
    """
    rules = []
    for i in range(st):
        for j in range(st):
            rules.append(((l_state, i, j), 0))
            rules.append(((((l_state - 1) % st), i, j), 1))
    return rules
