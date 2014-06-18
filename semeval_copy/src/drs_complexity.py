#!/usr/bin/env python
    
import sys
from xml.sax import parseString, handler

class FancyCounter(handler.ContentHandler):

    def __init__(self):
        self.level = 0
        self.leafLevels = []
        
    def startElement(self, name, attrs):
        if name in ["drs", "sdrs"]:
            self.level += 1
            self.last = "open"
            
    def endElement(self, name):
        if name in ["drs", "sdrs"]:
            self.level -= 1
            if self.last == "open":
                self.leafLevels.append(self.level)
            self.last = "close"

    def endDocument(self):
        if len(self.leafLevels) != 0:
            self.complexity = round(float(sum(self.leafLevels))/float(len(self.leafLevels)),2)
        else:
            print "error here"
            self.complexity = 0.0


# remove DTD declaration because the link is broken in GMB   
def parse_xml(lines):
    xml = ""
    for line in [lines.split('\n')[0]]+lines.split('\n')[1:]:
       # print line
        if line[:10] != "<!DOCTYPE ":
            xml += line+'\n'

    fc = FancyCounter()
    parseString(xml,fc)
    return fc.complexity

if __name__ == '__main__':
    xml = ""
    for line in sys.stdin.readlines():
        if line[:10] != "<!DOCTYPE ":
            xml += line

    print xml
    fc = FancyCounter()
    parseString(xml,fc)
    print fc.complexity



"""
4957 0.0 1.0
4959 1.0 0.0
error here
4960 0.0 0.0
error here
4961 0.5 0.0
4964 2.0 1.0
"""