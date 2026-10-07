from sklearn import tree
from collections import defaultdict
from progress.bar import IncrementalBar
import numpy as np
import pandas as pd
import xml.etree.ElementTree as et
import random
import string
import graphviz

class xmlDmn:
    
    def __init__(self, dmnName, outName = None):
        self.tree = et.parse(dmnName)
        self.xmlRoot = self.tree.getroot()   #uzima root element 
        self.namespace = self.xmlRoot.tag.split("}")[0].strip("{")  #iz cijelog taga uzima dio do "}" te uklanja "{", ostaje nam cisti xmlns
        self.ns = {"xmlns" : self.namespace} #ovaj dict je kreiran da se moze koristiti za potrebe find i findall
        self.decisionTableElement = self.xmlRoot.find(".//xmlns:decisionTable", namespaces=self.ns)  #pronalazi Decision Table element u kojem su spremljeni svi inputi i outputi 

    def printDecisionTable(self):
        ## Ispisuje decision table - ovo je vise radi debugganja
        for element in self.decisionTableElement:
            if self.namespace in element.tag:
                print(element.tag.split("}")[-1], ":", element.attrib)
    
    def idGen(self, text):
        #Jednostavni ID generator za potrebe id atributa
        return text + ''.join(random.choices(string.ascii_lowercase + string.digits, k=7))

    def clearDecisionTable(self):
        ## Cisti nepotrebne/stare elemente - time dobivamo cistu tablicu spremnu za upis novih podataka
        for element in self.xmlRoot.findall(".//xmlns:decisionTable/", namespaces=self.ns):
            if element.tag == "{%s}input"%(self.namespace):
                self.decisionTableElement.remove(element)
            elif element.tag == "{%s}output"%(self.namespace):
                self.decisionTableElement.remove(element)
            ## Cisti stara pravila
            elif element.tag == "{%s}rule"%(self.namespace):
                self.decisionTableElement.remove(element)
    
    def generateTableColumns(self,names):
        outputName = names[-1]
        inputNames = names[:-1]
        ## Kreiranje novih elementa
        for name in inputNames:
            newInput = et.SubElement(self.decisionTableElement, "{%s}input"%(self.namespace), attrib= {"id": self.idGen("input_")})
            newInputExpression = et.SubElement(newInput, "{%s}inputExpression"%(self.namespace), attrib={"id": self.idGen("inputExpression_"),"typeRef":"double"}) 
            newText = et.SubElement(newInputExpression, "{%s}text"%(self.namespace))
            newText.text = name
        et.SubElement(self.decisionTableElement, "{%s}output"%(self.namespace), attrib={"id": self.idGen("output_"),"name": outputName,"typeRef":"string"})

    def generateTableRows(self, mlDict):
        for k, v in mlDict:
            newRule = et.SubElement(self.decisionTableElement, "{%s}rule"%(self.namespace), attrib={"id":self.idGen("DecisionRule_")})
            for keyValues in v:
                tSignList = list(v[keyValues].keys())
                if len(v[keyValues]) == 0:
                    self.createRuleCell(newRule)               
                if len(v[keyValues]) == 1:    
                    self.createRuleCell(newRule).text = "{} {:.4f}".format(tSignList[0],v[keyValues][tSignList[0]])
                if len(v[keyValues]) == 2:
                    if tSignList[0] == "<=":
                        if v[keyValues][tSignList[0]] > v[keyValues][tSignList[1]]:
                            self.createRuleCell(newRule).text = "]{:.4f}..{:.4f}]".format(v[keyValues][tSignList[1]],v[keyValues][tSignList[0]])
                        else:
                            self.createRuleCell(newRule)
                    if tSignList[0] == ">":
                        if v[keyValues][tSignList[0]] < v[keyValues][tSignList[1]]:
                            self.createRuleCell(newRule).text = "]{:.4f}..{:.4f}]".format(v[keyValues][tSignList[0]],v[keyValues][tSignList[1]])
                        else:
                            self.createRuleCell(newRule)                    
            newOutEntry = et.SubElement(newRule, "{%s}outputEntry"%(self.namespace), attrib={"id":self.idGen("LiteralExpression_")})
            newOutText = et.SubElement(newOutEntry, "{%s}text"%(self.namespace))
            newOutText.text = str(k)


    def writeTree(self, outName=None):
        ## Ispis/spremanje u novi file
        if outName is None:
            self.tree.write('new.dmn')
        else:
            self.tree.write(outName)

class clfDmn(xmlDmn):

    def __init__(self,csvName, dmnName, target = None, depth = None, outName = None):
        super().__init__(dmnName)   #poziva konstruktor od "xmlDmn"
        df = pd.read_csv(csvName)   #kreira data frame
        self.dfColumns = df.columns
        if target is None:
            self.dfTarget = df[df.columns[-1]]  #dataframe u kojem se nalazi target odnosno klasa/karakteristika koju trazimo
            self.dfData = df[df.columns[:-1]]   #dataframe u kojem su podaci 
            self.dfFeature = self.dfColumns[:-1]    #sadrzi nazive inputa(karakteristike)
            self.dfClassColumn = self.dfColumns[-1] #sadrzi naziv targeta(klase koju trazimo)
        else:
            self.dfTarget = df[df.columns[target]]
            self.dfData = df[df.columns.drop(df.columns[target])]
            self.dfFeature = self.dfColumns.drop(self.dfColumns[target])
            self.dfClassColumn = self.dfColumns[target]
        self.dfClassNames = self.dfTarget.unique()
        if depth is None:
            #ovdje poziva classifier 
            self.clf = tree.DecisionTreeClassifier().fit(self.dfData,self.dfTarget)
        else:
            self.clf = tree.DecisionTreeClassifier(max_depth=depth).fit(self.dfData,self.dfTarget)
        print("Accuaracy on the training set : {:.3f}".format(self.clf.score(self.dfData,self.dfTarget)))
        
    def visualizeTree(self, name):
        #Kreira vizualizaciju stabla, pdf sa imenom "name"
        dot_data = tree.export_graphviz(self.clf, out_file=None, feature_names=self.dfFeature.astype(str), class_names=self.dfClassNames.astype(str), filled=True, rounded=True, special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render(name)
    
    def generateTableFromClf(self):
        featureNames = [self.dfFeature[i] for i in self.clf.tree_.feature]
        leafIds = self.clf.apply(self.dfData) 
        leftChildren = self.clf.tree_.children_left    
        rightChildren = self.clf.tree_.children_right 
        decPath = self.clf.decision_path(self.dfData)  
        threshold = self.clf.tree_.threshold   

        featuresForDiagram = list(self.dfFeature)   
        featuresForDiagram.append(self.dfClassColumn)  
        
        self.clearDecisionTable()  
        self.generateTableColumns(featuresForDiagram)

        bar = IncrementalBar("Processing",suffix='%(percent)d%%', max = len(set(leafIds)))  
        for i in set(leafIds): 
            samplesInNode = decPath.getcol(i).copy()   
            rows,cols = samplesInNode.nonzero()[0] 
            sampleId = rows[0] 
            nodeIndex = decPath.indices[decPath.indptr[sampleId]:decPath.indptr[sampleId+1]] 
            className = self.clf.classes_[np.argmax(self.clf.tree_.value[i])]
            inputOutput = defaultdict(dict)  
            for value in featuresForDiagram[:-1]:  
                    inputOutput[className][value] = {} 
            for index, nodeId in enumerate(nodeIndex):
                nodeFeature = featureNames[nodeIndex[index-1]] 
                nodeThreshold = threshold[nodeIndex[index-1]] 
                if nodeId in set(leftChildren):
                    inputOutput[className][nodeFeature]["<="] = nodeThreshold
                if nodeId in set(rightChildren):
                    inputOutput[className][nodeFeature][">"] = nodeThreshold
            self.generateTableRows(inputOutput.items())   
            bar.next()
        bar.finish()
        self.writeTree()   



if __name__ == "__main__":
    newObject = clfDmn(csvName="iris.csv", dmnName="flowers.dmn")
    #newObject = clfDmn(csvName="winequality-red.csv", dmnName="test.dmn")
    #newObject = clfDmn(csvName="letters.csv", dmnName="test.dmn",target = 0, depth=15)
    #newObject.visualizeTree("VisualizedTree")
    newObject.generateTableFromClf()