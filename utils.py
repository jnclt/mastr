#!/usr/bin/env python3
#encoding: utf-8
from os import mkdir
from os.path import isdir

import model


# all rules used in all forests
usedRules = {}
# all perceptions used in the rules in all forests
usedPerceptions = {}
# all system states used in all forests
usedStates = {}

def wTran_1(l_state):
    rules = model.tran_1(l_state)
    if not rules:
        rules = []
    # rules.sort()
    return set([getRefRule(rule) for rule in rules])

def getRefRule(rule):
    if rule not in usedRules:
        if rule[0] not in usedPerceptions:
            usedPerceptions[rule[0]] = rule[0]
        usedRules[rule] = (usedPerceptions[rule[0]], rule[1]);
    return usedRules[rule];

def getRefState(state):
    if state not in usedStates:
        usedStates[state] = state
    return usedStates[state]


strategyPath = "./strategy.txt"

def outputPath():
    """ Make sure the output directory exists and return its path.
    """
    output = "./output/"
    if not isdir(output):
        mkdir(output)
    return output


graphHeader = {}
graphFooter = {}
graphHeader["dot"] = \
"digraph untitled {\n"
graphFooter["dot"] = "}"
graphHeader["GraphML"] = \
'<?xml version="1.0" encoding="UTF-8"?>\n\
<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n\
xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n\
xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n\
http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n\
    <key id="l" for="node" attr.name="label" attr.type="string" />\n\
    <key id="c" for="node" attr.name="color" attr.type="string" >\n\
    \t<default>yellow</default>\n\
    </key>\n\
    <key id="e" for="edge" attr.name="elabel" attr.type="string" />\n\
    <graph id="G" edgedefault="directed">\n'
graphFooter["GraphML"] = \
'  </graph>\n\
</graphml>'
