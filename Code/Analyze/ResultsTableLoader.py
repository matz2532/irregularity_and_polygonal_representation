import numpy as np
import pandas as pd
import seaborn as sns
import sys

class ResultsTableLoader (object):

    # default conversion dicts (provided key, returns 'converted' value)
    colorPalette = sns.color_palette("colorblind")
    genotypeColorConversion = {"WT": colorPalette[7], "WT+Oryzalin": colorPalette[8], "$\it{ktn1}$-$\it{2}$": colorPalette[0]}
    labelNameConverterDict = {"lengthGiniCoeff":"Gini coefficient of length", "angleGiniCoeff":"Gini coefficient of angle",
                              "ratio_originalPolygonArea_labelledImageArea":"polygonal cell area / pixelized cell area",
                              "ratio_regularPolygonArea_labelledImageArea":"regularized polygonal cell area / pixelized cell area",
                              "ratio_regularPolygonArea_originalPolygonArea":"regularized polygonal cell area / polygonal cell area"}
    # default column names
    numericTimePointColName="time point numeric"
    scenarioColName="genotype" # due to old code this remains genotype even though it's sometimes scenario
    pureGenotypeColName="pure genotype" # actually named as pure genotype as genotype was actually used to indicate scenario
    replicateIdColName="replicate id"
    timePointNameColName="time point"
    tissueColName="tissue name"
    cellLabelColName="cell label"
    # default initialized parameters
    columnsToMergeDtypeDict={scenarioColName : str, replicateIdColName : str, "timePointNameColName" : str}
    filenamesOrTables=None
    combinedColumnName=None
    combinedColumnTypeName=None
    resultsTable=None

    def __init__(self, filenamesOrTables=None, timePointConversionFromStrToNumber=None, renameGenotypesDict=None,
                 renameTimePointDict=None, onlyShowLastAndFirstTimePoint=False, furtherRemoveScenarioIds=None, reduceResultsToOnePerTissue=False,
                 defaultColumnsToMergeOn=None):
        if defaultColumnsToMergeOn is None:
            self.defaultColumnsToMergeOn = [self.scenarioColName, self.timePointNameColName, self.replicateIdColName, self.cellLabelColName]
        else:
            self.defaultColumnsToMergeOn = defaultColumnsToMergeOn
        self.entryIdentifierColumns = [self.scenarioColName, self.replicateIdColName, self.timePointNameColName]
        if not filenamesOrTables is None:
            self.SetResultsTable(filenamesOrTables, timePointConversionFromStrToNumber=timePointConversionFromStrToNumber, renameGenotypesDict=renameGenotypesDict,
                                 renameTimePointDict=renameTimePointDict, onlyShowLastAndFirstTimePoint=onlyShowLastAndFirstTimePoint,
                                 furtherRemoveScenarioIds=furtherRemoveScenarioIds, reduceResultsToOnePerTissue=reduceResultsToOnePerTissue, defaultColumnsToMergeOn=self.defaultColumnsToMergeOn)

    def SetResultsTable(self, resultsTableFilenames, timePointConversionFromStrToNumber=None, renameGenotypesDict=None,
                        renameTimePointDict=None, onlyShowLastAndFirstTimePoint=False, furtherRemoveScenarioIds=None,
                        reduceResultsToOnePerTissue=False, defaultColumnsToMergeOn=None):
        self.resultsTableFilenames = resultsTableFilenames
        self.resultsTable = self.loadFilenamesOrTables(resultsTableFilenames, defaultColumnsToMergeOn=defaultColumnsToMergeOn)
        if not renameTimePointDict is None:
            self.resultsTable = self.resultsTable.replace({self.timePointNameColName:renameTimePointDict})
        if not timePointConversionFromStrToNumber is None:
            self.addTimePointAsNumber(self.resultsTable, timePointConversionFromStrToNumber, timePointColName=self.timePointNameColName)
        if not renameGenotypesDict is None:
            self.resultsTable = self.resultsTable.replace({self.scenarioColName: renameGenotypesDict})
        self.splitScenarioInTissueAndGeneName(self.resultsTable)
        if onlyShowLastAndFirstTimePoint:
            self.onlyKeepFirstAndLastTimePointsScenariosData()
        if not furtherRemoveScenarioIds is None:
            self.removeScenarioIdsData(furtherRemoveScenarioIds)
        if reduceResultsToOnePerTissue:
            self.removeAllExceptFirstUniqueScenarioData()

    def SetLabelNameConverterDict(self, labelNameConverterDict, extend=False):
        if extend:
            for k, v in labelNameConverterDict.items():
                self.labelNameConverterDict[k] = v
        else:
            self.labelNameConverterDict = labelNameConverterDict

    def GetCombinedColumnName(self):
        return self.combinedColumnName

    def GetExistingEntryIdentifiers(self, entryIdentifierColumns=None, sort=True):
        if entryIdentifierColumns is None:
            entryIdentifierColumns = self.entryIdentifierColumns
        allUniqueIds = []
        for ids, _ in self.resultsTable.groupby(entryIdentifierColumns, sort=sort):
            allUniqueIds.append(ids)
        return allUniqueIds

    def GetResultsTable(self):
        return self.resultsTable

    def GetResultsTableOf(self, entryIdentifier, entryIdentifierColumns=None):
        if entryIdentifierColumns is None:
            entryIdentifierColumns = self.entryIdentifierColumns
        # for ids, _ in self.resultsTable.groupby(entryIdentifierColumns):
        #             print(ids)
        # print(entryIdentifier)
        print(np.sum(self.resultsTable[entryIdentifierColumns] == entryIdentifier, axis=1).max())
        isEntry = len(entryIdentifierColumns) == np.sum(self.resultsTable[entryIdentifierColumns] == entryIdentifier, axis=1)
        return self.resultsTable[isEntry]

    def loadFilenamesOrTables(self, filenamesOrTables, defaultColumnsToMergeOn=None):
        if defaultColumnsToMergeOn is None:
            self.defaultColumnsToMergeOn = defaultColumnsToMergeOn
        if type(filenamesOrTables) == list:
            tables = []
            for filenameOrTable in filenamesOrTables:
                if type(filenameOrTable) == list:
                    currentTable = None
                    for nestedFilenameOrTable in filenameOrTable:
                        currentNestedTable = self.loadTables(nestedFilenameOrTable)
                        if currentTable is None:
                            currentTable = currentNestedTable
                        else:
                            currentTable = currentTable.merge(currentNestedTable, on=defaultColumnsToMergeOn)
                else:
                    currentTable = self.loadTables(filenameOrTable)
                tables.append(currentTable)
            table = pd.concat(tables).reset_index(drop=True)
        elif isinstance(filenamesOrTables, str):
            table = self.loadTables(filenamesOrTables)
        elif isinstance(filenamesOrTables, pd.DataFrame):
            table = filenamesOrTables
        else:
            raise NotImplementedError(f"Setting the current table using filenameOrTables are of type {type(filenamesOrTables)} != list, str, or pd.DataFrame")
        return table

    def loadTables(self, filenameOrTable):
        if isinstance(filenameOrTable, str):
            return pd.read_csv(filenameOrTable)
        elif isinstance(filenameOrTable, pd.DataFrame):
            return filenameOrTable
        else:
            raise NotImplementedError(f"Loading the current filenameOrTable of type {type(filenameOrTable)} != str or pd.DataFrame")

    def StackColumnsOfResultsTableVertically(self, namesOfColumnsToCombine, newColumnName="Ratio of areas", typeDenotingColumnExtension="_type"):
        assert not self.resultsTable is None, "The results table is not yet defined."
        isNamesColumnOfTypeDict = [isinstance(columnName, dict) for columnName in namesOfColumnsToCombine]
        namesOfColumnsToCombine = np.array(namesOfColumnsToCombine)
        idxOfColumnsOfTypeDict = np.where(isNamesColumnOfTypeDict)[0]
        for i in idxOfColumnsOfTypeDict:
            dictColName = namesOfColumnsToCombine[i]
            assert "measureCol" in dictColName, f"The key 'measureCol' is missing in the {dictColName=} of {namesOfColumnsToCombine=}, when combining columns to result table"
            currentColName = dictColName["measureCol"]
            if "inverse" in dictColName:
                doInverse = dictColName["inverse"]
                if doInverse:
                    currentNewColName = "inverse_" + currentColName
                    self.resultsTable[currentNewColName] = 1 / self.resultsTable[currentColName]
                    currentColName = currentNewColName
            namesOfColumnsToCombine[i] = currentColName
        columns = self.resultsTable.columns
        isColumnPresent = np.isin(namesOfColumnsToCombine, columns)
        assert np.all(isColumnPresent), f"The columns {columns[np.invert(isColumnPresent)].to_list()} to combine are not present in the columns of the results table, select one fo the following {columns}"
        numberOfResults = len(self.resultsTable)
        columnsToInclude = [*columns[:4], self.tissueColName, self.pureGenotypeColName, self.numericTimePointColName, *namesOfColumnsToCombine]
        resultsTable = [pd.DataFrame(self.resultsTable[columnsToInclude]) for _ in namesOfColumnsToCombine]
        self.combinedColumnName = newColumnName
        self.combinedColumnTypeName = newColumnName + typeDenotingColumnExtension
        for i, columnName in enumerate(namesOfColumnsToCombine):
            newColumnValues = self.resultsTable[columnName]
            columnTypeValues = numberOfResults * [columnName]
            resultsTable[i][newColumnName] = newColumnValues
            resultsTable[i][self.combinedColumnTypeName] = columnTypeValues
        self.resultsTable = resultsTable

    def addTimePointAsNumber(self, resultsTable, timePointConversionFromStrToNumber, timePointColName="time point", timePointDtype=int):
        numberOfCells = len(resultsTable)
        allNumericTimePointValues = np.zeros(numberOfCells, dtype=timePointDtype)
        strTimePointValues = resultsTable[timePointColName]
        for i, strTimePoint in enumerate(strTimePointValues):
            if strTimePoint in timePointConversionFromStrToNumber:
                numericTimePointValue = timePointConversionFromStrToNumber[strTimePoint]
            else:
                print(f"The time point with name {strTimePoint} does not exist in the string to numerical value conversion dictionary {timePointConversionFromStrToNumber}")
                sys.exit()
            allNumericTimePointValues[i] = numericTimePointValue
        resultsTable[self.numericTimePointColName] = allNumericTimePointValues

    def splitScenarioInTissueAndGeneName(self, resultsTable):
        allGenNames, allTissueNames = [], []
        genotypeColumn = self.resultsTable[self.scenarioColName]
        for i, genotype in enumerate(genotypeColumn):
            splitGenotype = genotype.split(" ")
            geneName = splitGenotype[0]
            tissueName = " ".join(splitGenotype[1:])
            allGenNames.append(geneName)
            allTissueNames.append(tissueName)
        resultsTable[self.pureGenotypeColName] = allGenNames
        resultsTable[self.tissueColName] = allTissueNames

    def onlyKeepFirstAndLastTimePointsScenariosData(self):
        nonFirstOrLastRowIndices = []
        for scenarioId, scenarioTable in self.resultsTable.groupby([self.scenarioColName], sort=False):
            timePoints = scenarioTable[self.timePointNameColName]
            uniqueTimePoints = timePoints.unique()
            if "h" in uniqueTimePoints[0] and not "-" in uniqueTimePoints[0]:
                uniqueTimePointsAsInt = [int(t[:-1]) for t in uniqueTimePoints]
                argsortTimePoints = np.argsort(uniqueTimePointsAsInt)
                for t in uniqueTimePoints:
                    argOfTimePoint = argsortTimePoints[t == uniqueTimePoints][0]
                    isLastOrFirst = argOfTimePoint == 0 or argOfTimePoint == np.max(argsortTimePoints)
                    if not isLastOrFirst:
                        rowsToRemove = timePoints.index[t == timePoints]
                        nonFirstOrLastRowIndices.extend(list(rowsToRemove))
        self.resultsTable = self.resultsTable.drop(nonFirstOrLastRowIndices, axis=0)

    def removeScenarioIdsData(self, furtherRemoveScenarioIds):
        rowIndicesToDrop = []
        for scenarioId, scenarioTable in self.resultsTable.groupby([self.tissueColName, self.pureGenotypeColName, self.timePointNameColName], sort=False):
            isScenarioInList = self.findIfScenarioIdIsAmong(scenarioId, furtherRemoveScenarioIds)
            if isScenarioInList:
                rowsToRemove = list(scenarioTable.index)
                rowIndicesToDrop.extend(rowsToRemove)
        self.resultsTable = self.resultsTable.drop(rowIndicesToDrop, axis=0)

    def findIfScenarioIdIsAmong(self, scenarioId, furtherRemoveScenarioIds):
        for possibleId in furtherRemoveScenarioIds:
            if scenarioId == possibleId:
                return True
        return False

    def removeAllExceptFirstUniqueScenarioData(self):
        reducedResultsTable = []
        for scenarioId, scenarioTable in self.resultsTable.groupby([self.tissueColName, self.pureGenotypeColName, self.timePointNameColName, self.replicateIdColName], sort=False):
            reducedResultsTable.append(scenarioTable.iloc[0, :])
        self.resultsTable = pd.concat(reducedResultsTable, axis=1).T

def checkCombiningOfMultipleTables():
    # values to be set
    measureName = "lengthGiniCoeff"
    tissueScenarioNames = ["", "full cotyledons", "SAM"]
    baseResultsFolder = "Results/"
    measuresBaseName = "combinedMeasures.csv"
    saveFigure = True
    resultsTableFilenames = [f"{baseResultsFolder}{i}/{measuresBaseName}" if i != "" else f"{baseResultsFolder}{measuresBaseName}" for i in tissueScenarioNames]

    # values for ordering data
    tissueGeneNameOrdering = {"SAM": ["WT"], "cotyledon": ["WT", "WT+Oryzalin", "$\it{ktn1}$-$\it{2}$"]}
    renameTimePointDict = {'1DAI': "24h", '1.5DAI': "36h", '2DAI': "48h", '2.5DAI': "60h", '3DAI': "72h", '3.5DAI': "84h", '4DAI': "96h", '4.5DAI': "108h", '5DAI': "120h", '5.5DAI': "132h", '6DAI': "144h",
                           "11h": "12h", "35h": "36h", "45h": "48h", "57h": "60h", "69h": "72h", "25h": "24h", "50h": "48h", "77h": "72h", "22h": "24h"}
    timePointConversionFromStrToNumber = {"0h": 0, '12h': 12, "24h": 24, '36h': 36, "48h": 48, '60h': 60, "72h": 72, '84h': 84, "96h": 96, '108h': 108, "120h": 120, '132h': 132, '144h': 144, "T0": 0, "T1": 0, "T2": 0, "T3": 0}
    renameGenotypesDict = {"WT inflorescence meristem": "WT SAM", "col-0": "WT cotyledon", "WT": "WT cotyledon", "Oryzalin": "WT+Oryzalin cotyledon", "ktn1-2": "$\it{ktn1}$-$\it{2}$ cotyledon"}

    ResultsTableLoader(resultsTableFilenames)
    
if __name__ == '__main__':
    checkCombiningOfMultipleTables()