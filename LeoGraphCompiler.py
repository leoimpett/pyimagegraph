import numpy as np

import json
import urllib.parse

class LeoCompiler:
    def __init__(self, filepath=None, instring=None, injson=None):
        if filepath is not None:
            with open(filepath, 'rt') as myfile:
                instring = myfile.read()
                self.json_string = urllib.parse.unquote(instring)[29:]
                self.json_data = json.loads(self.json_string)
        elif instring is not None:
            self.json_string = urllib.parse.unquote(instring)[29:]
            self.json_data = json.loads(self.json_string)
        elif injson is not None:
            self.json_data =  injson
        else:
            raise('Needs JSON to reasonably init')
        self.mynodes = self.json_data['nodes']
        self.mylinks = self.json_data['links']
        




    def isReady(self, thisnode):
        is_ready = 1
        # Are all the inputs satisfied? If not, then the node is not yet 'ready' to be compiled - others should be done first
        input_ports = [p for p in self.mynodes[thisnode]['ports'] if self.mynodes[thisnode]['ports'][p]['type'] == 'top']
        for thisport in input_ports:
            if 'isComputed' not in self.mynodes[thisnode]['ports'][thisport]:
                is_ready = 0
        # If the outputs are ready, that means that the node is already computed! So it is not 'ready' for computation
        # in the sense that it's already been compiled
        output_ports = [p for p in self.mynodes[thisnode]['ports'] if self.mynodes[thisnode]['ports'][p]['type'] == 'bottom']
        for thisport in output_ports:
            if 'isComputed' in self.mynodes[thisnode]['ports'][thisport]:
                is_ready = 0
        return is_ready



    def setComputed(self, thisnode):
        output_vars = []
        #find all the bottom-facing ports..
        #note to self - how do we make sure they're in the right order?
        output_ports = [p for p in self.mynodes[thisnode]['ports'] if self.mynodes[thisnode]['ports'][p]['type'] == 'bottom']
        #for every output port
        for thisport in output_ports:
            output_vars.append(self.mynodes[thisnode]['ports'][thisport]['variable'])

        for output_var in output_vars:
            for node in self.mynodes:
                for port in self.mynodes[node]['ports']:
                    if 'variable' in self.mynodes[node]['ports'][port]:
                        if self.mynodes[node]['ports'][port]['variable'] == output_var:
                            self.mynodes[node]['ports'][port]['isComputed'] = True

        # note that it might also be an input-only box, which means that there won't be any output ports to set to 0
        # in that case, let's make a fake port...
        if len(output_vars) == 0:
            self.mynodes[thisnode]['ports']['fakeport'] = {}
            self.mynodes[thisnode]['ports']['fakeport']['type'] = 'bottom'
            self.mynodes[thisnode]['ports']['fakeport']['isComputed'] = True


    def getOutputVars(self, thisnode):
        output_vars = ''
        #find all the bottom-facing ports..
        #note to self - how do we make sure they're in the right order?
        output_ports = [p for p in self.mynodes[thisnode]['ports'] if self.mynodes[thisnode]['ports'][p]['type'] == 'bottom']
        #for every output port
        for thisport in output_ports:
            #if it has a variable assigned (some may be optional)
            if 'variable' in self.mynodes[thisnode]['ports'][thisport]:
                output_vars += self.mynodes[thisnode]['ports'][thisport]['variable']
            else:
                # if there isn't a variable name assigned yet, it may be optional - in this case put an _
                output_vars += '_ignore '
            output_vars += ','
        if len(output_ports) == 0:
            output_vars = '_ '
        return output_vars[:-1]



    def getInputVars(self, thisnode):
        input_vars = ''
        #find all the top-facing ports..
        #note to self - how do we make sure they're in the right order? Maybe input_ports.sort() ?
        input_ports = [p for p in self.mynodes[thisnode]['ports'] if self.mynodes[thisnode]['ports'][p]['type'] == 'top']
        #for every input port
        for thisport in input_ports:
            #if it has a variable assigned (some may be optional)
            if 'variable' in self.mynodes[thisnode]['ports'][thisport]:
                input_vars += self.mynodes[thisnode]['ports'][thisport]['variable']
            else:
                # if there isn't a variable name assigned yet, it may be optional - in this case put a None
                input_vars += 'None '
            input_vars += ','
        if len(input_ports)>0:
            return input_vars[:-1]
        #if there aren't any input ports, then return the internal value, as a string (everything should be passed as str)
        else:
            #note that it also needs some speech bubbles to make it parse as an explicit string:
            return '"' + self.mynodes[thisnode]['properties']['innerValue'] + '"'
    
    
    
    def genVarNames(self):
        # The first thing we want to do is automatically generate a variable name for each output-port!
        # NB you can't have various things going into the same port, but you can have various things coming out. 
        k = 0
        for thisnode in self.mynodes:
            output_ports = [p for p in self.mynodes[thisnode]['ports'] if self.mynodes[thisnode]['ports'][p]['type'] == 'bottom']

            for thisport in output_ports:
                dataType = self.mynodes[thisnode]['ports'][thisport]['properties']['type']
                varName = dataType + str(k).zfill(2)
                k += 1
                self.mynodes[thisnode]['ports'][thisport]['variable'] = varName

    def linkVariables(self):
        # Then we want to assign these variable names to the nodes ; to each input/output name. 
        for lk in self.mylinks:
            # Check the nodes are actually connected and not just freehanging...
            if 'nodeId' in self.mylinks[lk]['from'] and 'nodeId' in self.mylinks[lk]['to']:
                node_A = self.mylinks[lk]['from']['nodeId']
                port_A = self.mylinks[lk]['from']['portId']
                node_B = self.mylinks[lk]['to']['nodeId']
                port_B = self.mylinks[lk]['to']['portId']

                # Set the variable name of the input to box B as the name of the output to box A, given that we're 
                # creating variable names by outputs
                self.mynodes[node_B]['ports'][port_B]['variable'] = self.mynodes[node_A]['ports'][port_A]['variable']
            
            
    def writePython(self):
        self.python_string = []

        self.python_string.append('from LeoGraphLib import * \n')
        # the next thing we want to do is fetch a list of nodes without unsatisfied inputs:

        while np.any([self.isReady(node) for node in self.mynodes]):
            readyNodes = [node for node in self.mynodes if self.isReady(node)]


            # select the first one, and generate the list of outputs and the list of inputs:

            outputs = self.getOutputVars(readyNodes[0])

            inputs = self.getInputVars(readyNodes[0])

            self.python_string.append(outputs + ' = ' + self.mynodes[readyNodes[0]]['type'] + '(' + inputs  +  ') \n')

            # set isComputed to true in all ports which use these outputs
            self.setComputed(readyNodes[0])
            
        with open('./tmpexec.py', 'w') as outfile:
            for L in self.python_string:
                outfile.writelines(L) 
                
    def compileGraph(self):
        self.genVarNames()
        self.linkVariables()
        self.writePython()
        print('Compiled graph')
            
    
            