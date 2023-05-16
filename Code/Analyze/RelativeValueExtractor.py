import numpy as np
from ExtendedPlotter import ExtendedPlotter

class RelativeValueExtractor (ExtendedPlotter):

    def __init__(self, resultsTableFilenames, timePointConversionFromStrToNumber=None, renameGenotypesDict=None, renameTimePointDict=None, onlyShowLastAndFirstTimePoint=False, furtherRemoveScenarioIds=None):
        super().__init__(resultsTableFilenames, timePointConversionFromStrToNumber, renameGenotypesDict, renameTimePointDict, onlyShowLastAndFirstTimePoint, furtherRemoveScenarioIds=furtherRemoveScenarioIds)

    def CompareRelativeChangesBetween(self, scenarioIdFrom, scenarioIdTo, measureName, groupByTissueName=True, groupByGene=True, groupByTimePoint=True, printOut=False):
        assert len(scenarioIdFrom) == len(scenarioIdTo), f"The scenario ids {scenarioIdFrom=}, {scenarioIdTo=} don't have the same length. {len(scenarioIdFrom)} != {len(scenarioIdTo)}"
        assert np.sum([groupByTissueName, groupByGene, groupByTimePoint]) == len(scenarioIdFrom), f"The selected groupings ({groupByTissueName=}, {groupByGene=}, {groupByTimePoint=}) do not correspond to the {scenarioIdFrom=}. {np.sum([groupByTissueName, groupByGene, groupByTimePoint])} != {len(scenarioIdFrom)}"
        groupBy = self.selectGroupings(groupByTissueName, groupByGene, groupByTimePoint)
        meanValueOfScenarioFrom, meanValueOfScenarioTo = None, None
        for scenarioId, scenarioTable in self.resultsTable.groupby(groupBy, sort=False):
            if scenarioId == scenarioIdFrom:
                meanValueOfScenarioFrom = scenarioTable[measureName].mean()
            if scenarioId == scenarioIdTo:
                meanValueOfScenarioTo = scenarioTable[measureName].mean()
        if not meanValueOfScenarioFrom is None and not meanValueOfScenarioTo is None:
            relativeChange = self.calcRelativeChanges(meanValueOfScenarioFrom, meanValueOfScenarioTo)
            if printOut:
                print(relativeChange)
            return relativeChange
        allScenarioIds = [i for i, _ in list(self.resultsTable.groupby(groupBy, sort=False))]
        if meanValueOfScenarioFrom is None and meanValueOfScenarioTo is None:
            print(f"{scenarioIdFrom=} and {scenarioIdTo=} have not been found among the scenarios: {allScenarioIds}")
        elif meanValueOfScenarioFrom is None:
            print(f"{scenarioIdFrom=} has not been found among the scenarios: {allScenarioIds}")
        elif meanValueOfScenarioTo is None:
            print(f"{scenarioIdTo=} has not been found among the scenarios: {allScenarioIds}")
        return None

    def ExtractMeanValueOverScenarios(self, measureName, groupByTissueName=True, groupByGene=True, groupByTimePoint=True):
        groupBy = self.selectGroupings(groupByTissueName, groupByGene, groupByTimePoint)
        meanValuesOfScenarios = []
        for scenarioId, scenarioTable in self.resultsTable.groupby(groupBy, sort=False):
            meanValue = scenarioTable[measureName].mean()
            meanValuesOfScenarios.append(meanValue)
        return np.mean(meanValuesOfScenarios)

    def ExtractRelativeChanges(self, ofColumnName, vsValue, groupByTissueName=True, groupByGene=True, groupByTimePoint=True, argsort=True):
        groupBy = self.selectGroupings(groupByTissueName, groupByGene, groupByTimePoint)
        relativeChangesOf = {}
        for scenarioId, scenarioTable in self.resultsTable.groupby(groupBy, sort=False):
            meanValue = scenarioTable[ofColumnName].mean()
            relativeChange = self.calcRelativeChanges(vsValue, meanValue)
            relativeChangesOf[scenarioId] = relativeChange
        if argsort:
            relativeChangesOf = self.sortDictByValues(relativeChangesOf)
        return relativeChangesOf

    def selectGroupings(self, groupByTissueName=True, groupByGene=True, groupByTimePoint=True):
        groupBy = []
        if groupByTissueName:
            groupBy.append(self.tissueColName)
        if groupByGene:
            groupBy.append(self.pureGenotypeColName)
        if groupByTimePoint:
            groupBy.append(self.timePointNameColName)
        return groupBy

    def calcRelativeChanges(self, startValue, endValue, inPercent=True):
        calcRelChange = (endValue - startValue) / startValue
        if inPercent:
            calcRelChange *= 100
        return calcRelChange

    def sortDictByValues(self, dictWithFloatValues, convertKeyToTuple=True):
        tmpChange = {}
        keys = np.array(list(dictWithFloatValues.keys()))
        values = np.array(list(dictWithFloatValues.values()))
        argSort = np.argsort(np.abs(values))[::-1]
        for sortIdx in argSort:
            value = values[sortIdx]
            key = keys[sortIdx]
            if convertKeyToTuple:
                key = tuple(key)
            tmpChange[key] = value
        return tmpChange

def mainPrintMeanValueOverScenarios(measureName="lengthGiniCoeff", tissueScenarioNames=["", "full cotyledons", "SAM", "first leaf"],
                                    overwriteTimePointRenamingWith=None, furtherRemoveScenarioIds=None,
                                    baseResultsFolder="Results/", measuresBaseName="combinedMeasures.csv", onlyShowLastAndFirstTimePoint=False):
    resultsTableFilenames = [f"{baseResultsFolder}{i}/{measuresBaseName}" if i != "" else f"{baseResultsFolder}{measuresBaseName}" for i in tissueScenarioNames]
    renameTimePointDict = {'1DAI': "24h", '1.5DAI': "36h", '2DAI': "48h", '2.5DAI': "60h", '3DAI': "72h", '3.5DAI': "84h", '4DAI': "96h", '4.5DAI': "108h", '5DAI': "120h", '5.5DAI': "132h", '6DAI': "144h",
                           "11h": "12h", "35h": "36h", "45h": "48h", "57h": "60h", "69h": "72h", "25h": "24h", "50h": "48h", "77h": "72h", "22h": "24h"}
    if not overwriteTimePointRenamingWith is None:
        for timePointName, renameTo in overwriteTimePointRenamingWith.items():
            renameTimePointDict[timePointName] = renameTo
    timePointConversionFromStrToNumber = {"0h":0, '12h':12, "24h":24, '36h':36, "48h":48, '60h':60, "72h":72, '84h':84, "96h":96, '108h':108, "120h":120, '132h':132, '144h':144, "T0":0, "T1":24, "T2":48, "T3":72, "T4":96, "0-96h":0}
    renameGenotypesDict = {"WT inflorescence meristem":"WT SAM", "col-0":"WT cotyledon", "WT":"WT cotyledon", "first_leaf_LeGloanec2022":"WT 1st leaf", "Oryzalin":"WT+Oryzalin cotyledon", "ktn1-2":"$\it{ktn1}$-$\it{2}$ cotyledon", "clasp-1":"$\it{clasp}$-$\it{1}$ cotyledon", "speechless_Fox2018_MGXfromLeGloanec":"$\it{speechless}$ 1st leaf"}
    myRelativeValueExtractor = RelativeValueExtractor(resultsTableFilenames, timePointConversionFromStrToNumber, renameGenotypesDict=renameGenotypesDict,
                                                      renameTimePointDict=renameTimePointDict, onlyShowLastAndFirstTimePoint=onlyShowLastAndFirstTimePoint,
                                                      furtherRemoveScenarioIds=furtherRemoveScenarioIds)
    meanValuesOfScenarios = myRelativeValueExtractor.ExtractMeanValueOverScenarios(measureName)
    print(f"mainPrintMeanValueOverScenarios of {measureName}")
    print(meanValuesOfScenarios)

def mainPrintRelativeDiffBetweenScenarios(scenarioIdFrom = ('1st leaf', 'WT', '48h'), scenarioIdTo = ('1st leaf', 'WT', '60h'),
                                          measureName="lengthGiniCoeff", tissueScenarioNames=["", "full cotyledons", "SAM", "first leaf"], overwriteTimePointRenamingWith=None,
                                          baseResultsFolder="Results/", measuresBaseName="combinedMeasures.csv", onlyShowLastAndFirstTimePoint=False, printOut=True):
    resultsTableFilenames = [f"{baseResultsFolder}{i}/{measuresBaseName}" if i != "" else f"{baseResultsFolder}{measuresBaseName}" for i in tissueScenarioNames]
    renameTimePointDict = {'1DAI': "24h", '1.5DAI': "36h", '2DAI': "48h", '2.5DAI': "60h", '3DAI': "72h", '3.5DAI': "84h", '4DAI': "96h", '4.5DAI': "108h", '5DAI': "120h", '5.5DAI': "132h", '6DAI': "144h",
                           "11h": "12h", "35h": "36h", "45h": "48h", "57h": "60h", "69h": "72h", "25h": "24h", "50h": "48h", "77h": "72h", "22h": "24h", "T0": "0h", "T1": "24h", "T2": "48h", "T3": "72h", "T4":"96h"} #, "T1": "0-96h", "T2": "0-96h", "T3": "0-96h", "T4": "0-96h"} #
    if not overwriteTimePointRenamingWith is None:
        for timePointName, renameTo in overwriteTimePointRenamingWith.items():
            renameTimePointDict[timePointName] = renameTo
    timePointConversionFromStrToNumber = {"0h":0, '12h':12, "24h":24, '36h':36, "48h":48, '60h':60, "72h":72, '84h':84, "96h":96, '108h':108, "120h":120, '132h':132, '144h':144, "T0":0, "T1":24, "T2":48, "T3":72, "T4":96, "0-96h":0}
    renameGenotypesDict = {"WT inflorescence meristem":"WT SAM", "col-0":"WT cotyledon", "WT":"WT cotyledon", "first_leaf_LeGloanec2022":"WT 1st leaf", "Oryzalin":"WT+Oryzalin cotyledon", "ktn1-2":"$\it{ktn1}$-$\it{2}$ cotyledon", "clasp-1":"$\it{clasp}$-$\it{1}$ cotyledon", "speechless_Fox2018_MGXfromLeGloanec":"$\it{speechless}$ 1st leaf"}
    myRelativeValueExtractor = RelativeValueExtractor(resultsTableFilenames, timePointConversionFromStrToNumber, renameGenotypesDict=renameGenotypesDict,
                                                      renameTimePointDict=renameTimePointDict, onlyShowLastAndFirstTimePoint=onlyShowLastAndFirstTimePoint)
    relativeChangesOf = myRelativeValueExtractor.CompareRelativeChangesBetween(scenarioIdFrom, scenarioIdTo, measureName, printOut=printOut)
    return relativeChangesOf

def mainPrintMultipleRelativeDiffBetweenScenarios(listOfScenarioIdFrom = [('1st leaf', 'WT', '48h'), ('1st leaf', 'WT', '48h'), ('1st leaf', 'WT', '48h'), ('1st leaf', '$\\it{speechless}$', '24h'), ('1st leaf', '$\\it{speechless}$', '48h')],
                                                  listOfScenarioIdTo = [('1st leaf', 'WT', '96h'), ('1st leaf', 'WT', '108h'), ('1st leaf', 'WT', '132h'), ('1st leaf', '$\\it{speechless}$', '84h'), ('1st leaf', '$\\it{speechless}$', '84h')],
                                                  measureName="lengthGiniCoeff", tissueScenarioNames=["", "full cotyledons", "SAM", "first leaf"], overwriteTimePointRenamingWith=None,
                                                  baseResultsFolder="Results/", measuresBaseName="combinedMeasures.csv", onlyShowLastAndFirstTimePoint=False):
    print("mainPrintMultipleRelativeDiffBetweenScenarios", measureName)
    for scenarioIdFrom, scenarioIdTo in zip(listOfScenarioIdFrom, listOfScenarioIdTo):
        changes = mainPrintRelativeDiffBetweenScenarios(scenarioIdFrom, scenarioIdTo, measureName=measureName, tissueScenarioNames=tissueScenarioNames,
                                                        baseResultsFolder=baseResultsFolder, measuresBaseName=measuresBaseName, overwriteTimePointRenamingWith=overwriteTimePointRenamingWith,
                                                        onlyShowLastAndFirstTimePoint=onlyShowLastAndFirstTimePoint, printOut=False)
        print(scenarioIdFrom, scenarioIdTo, changes)

def mainPrintRelativeDiffTo(valueToCheckAgainst, measureName="lengthGiniCoeff", tissueScenarioNames=["", "full cotyledons", "SAM", "first leaf"],
                            baseResultsFolder="Results/", measuresBaseName="combinedMeasures.csv", onlyShowLastAndFirstTimePoint=True, overwriteTimePointRenamingWith=None):
    resultsTableFilenames = [f"{baseResultsFolder}{i}/{measuresBaseName}" if i != "" else f"{baseResultsFolder}{measuresBaseName}" for i in tissueScenarioNames]
    renameTimePointDict = {'1DAI': "24h", '1.5DAI': "36h", '2DAI': "48h", '2.5DAI': "60h", '3DAI': "72h", '3.5DAI': "84h", '4DAI': "96h", '4.5DAI': "108h", '5DAI': "120h", '5.5DAI': "132h", '6DAI': "144h",
                           "11h": "12h", "35h": "36h", "45h": "48h", "57h": "60h", "69h": "72h", "25h": "24h", "50h": "48h", "77h": "72h", "22h": "24h", "T0": "0-96h", "T1": "0-96h", "T2": "0-96h", "T3": "0-96h", "T4": "0-96h"} #, "T0": "0h", "T1": "24h", "T2": "48h", "T3": "72h", "T4":"96h"} #
    if not overwriteTimePointRenamingWith is None:
        for timePointName, renameTo in overwriteTimePointRenamingWith.items():
            renameTimePointDict[timePointName] = renameTo
    timePointConversionFromStrToNumber = {"0h":0, '12h':12, "24h":24, '36h':36, "48h":48, '60h':60, "72h":72, '84h':84, "96h":96, '108h':108, "120h":120, '132h':132, '144h':144, "T0":0, "T1":24, "T2":48, "T3":72, "T4":96, "0-96h":0}
    renameGenotypesDict = {"WT inflorescence meristem":"WT SAM", "col-0":"WT cotyledon", "WT":"WT cotyledon", "first_leaf_LeGloanec2022":"WT 1st leaf", "Oryzalin":"WT+Oryzalin cotyledon", "ktn1-2":"$\it{ktn1}$-$\it{2}$ cotyledon", "clasp-1":"$\it{clasp}$-$\it{1}$ cotyledon", "speechless_Fox2018_MGXfromLeGloanec":"$\it{speechless}$ 1st leaf"}
    myRelativeValueExtractor = RelativeValueExtractor(resultsTableFilenames, timePointConversionFromStrToNumber, renameGenotypesDict=renameGenotypesDict,
                                                      renameTimePointDict=renameTimePointDict, onlyShowLastAndFirstTimePoint=onlyShowLastAndFirstTimePoint)
    relativeChangesOf = myRelativeValueExtractor.ExtractRelativeChanges(measureName, valueToCheckAgainst)
    print("mainPrintRelativeDiffTo")
    print(relativeChangesOf)
def printGeneralGDifferencesToIdentifyDirectionality():
    import pandas as pd
    baseReultsFolder, baseResultsName = "Results/", "combinedMeasures.csv"
    checkAgainstEachOther = [["lengthGiniCoeff generalG Value", "lengthGiniCoeff expected GeneralG", "lengthGiniCoeff generalG pValue"], ["angleGiniCoeff generalG Value", "angleGiniCoeff expected GeneralG", "angleGiniCoeff generalG pValue"]]
    tissueScenarioFolderNames = ["", "full cotyledons/", "SAM/", "first leaf/"]
    for i, j, k in checkAgainstEachOther:
        print("\n", i, j, k)
        for tissueFolder in tissueScenarioFolderNames:
            resultsFilename = baseReultsFolder + tissueFolder + baseResultsName
            df = pd.read_csv(resultsFilename)
            for groupId, groupDf in df.groupby(["genotype", "time point", "replicate id"]):
                value, expectedValue, pValue = groupDf[i].values[0], groupDf[j].values[0], groupDf[k].values[0]
                print(np.round(value, 3), np.round(value - expectedValue, 3), np.round(pValue, 2), groupId)


if __name__ == '__main__':
    mainPrintRelativeDiffTo(1, measureName="ratio_originalPolygonArea_labelledImageArea", tissueScenarioNames=["full cotyledons", "SAM"])
    # mainPrintRelativeDiffBetweenScenarios(measureName="ratio_originalPolygonArea_labelledImageArea")
    # mainPrintMultipleRelativeDiffBetweenScenarios(measureName="ratio_originalPolygonArea_labelledImageArea")
    # mainPrintMeanValueOverScenarios(measureName="ratio_regularPolygonArea_originalPolygonArea")
    mainPrintMultipleRelativeDiffBetweenScenarios(measureName="ratio_regularPolygonArea_originalPolygonArea", tissueScenarioNames=[""],
                                                  listOfScenarioIdFrom=[('cotyledon', 'WT', '0h'), ('cotyledon', 'WT', '0h')],
                                                  listOfScenarioIdTo=[('cotyledon', 'WT', '96h'), ('cotyledon', '$\\it{clasp}$-$\\it{1}$', '96h')])
    # mainPrintMultipleRelativeDiffBetweenScenarios(measureName="ratio_regularPolygonArea_originalPolygonArea",
    #                                               listOfScenarioIdFrom=[('SAM', 'WT', '0h')],
    #                                               listOfScenarioIdTo=[('SAM', 'WT', '96h')]
    #                                               )
    # mainPrintMultipleRelativeDiffBetweenScenarios(measureName="distance normalised by mean labelledImageArea", measuresBaseName="combinedMeasures_neighborDistance.csv",
    #                                               listOfScenarioIdFrom=[('SAM', 'WT', '96h'), ('SAM', 'WT', '96h')],
    #                                               listOfScenarioIdTo=[('cotyledon', 'WT', '120h'), ('1st leaf', 'WT', '144h')],
    #                                               overwriteTimePointRenamingWith={"T0": "0h", "T1": "24h", "T2": "48h", "T3": "72h", "T4":"96h"}
    #                                               )
    # mainPrintMultipleRelativeDiffBetweenScenarios(measureName="lengthGiniCoeff", measuresBaseName="combinedMeasures.csv",
    #                                               listOfScenarioIdFrom=[('SAM', 'WT', '0-96h'), ('SAM', 'WT', '0-96h'), ('1st leaf', 'WT', '24h'), ('1st leaf', '$\\it{speechless}$', '0h'), ('SAM', 'WT', '0-96h'), ('cotyledon', 'WT', '120h')],
    #                                               listOfScenarioIdTo=[('cotyledon', 'WT', '0h'), ('1st leaf', 'WT', '24h'), ('1st leaf', 'WT', '144h'), ('1st leaf', '$\\it{speechless}$', '96h'), ('cotyledon', '$\it{ktn1}$-$\it{2}$', '120h'), ('cotyledon', '$\it{ktn1}$-$\it{2}$', '120h')],
    #                                               overwriteTimePointRenamingWith={"T0": "0-96h", "T1": "0-96h", "T2": "0-96h", "T3": "0-96h", "T4":"0-96h"}
    #                                               )
    # mainPrintMultipleRelativeDiffBetweenScenarios(measureName="angleGiniCoeff", measuresBaseName="combinedMeasures.csv",
    #                                               listOfScenarioIdFrom=[('SAM', 'WT', '0-96h'), ('SAM', 'WT', '0-96h'), ('cotyledon', 'WT', '120h'), ('cotyledon', '$\it{clasp}$-$\it{1}$', '120h')],
    #                                               listOfScenarioIdTo=[('cotyledon', 'WT', '120h'), ('1st leaf', 'WT', '144h'), ('cotyledon', '$\it{ktn1}$-$\it{2}$', '120h'), ('cotyledon', 'WT', '120h')],
    #                                               overwriteTimePointRenamingWith={"T0": "0-96h", "T1": "0-96h", "T2": "0-96h", "T3": "0-96h", "T4":"0-96h"}
    #                                               )
    # mainPrintMultipleRelativeDiffBetweenScenarios(measureName="lengthGiniCoeff", measuresBaseName="combinedMeasures.csv",
    #                                               listOfScenarioIdFrom=[('SAM', 'WT', '0-96h'), ('1st leaf', 'WT', '144h'),],
    #                                               listOfScenarioIdTo=[('inflorescence meristem', 'ktn', '0-96h'), ('1st leaf', '$\\it{speechless}$', '96h'),],
    #                                               overwriteTimePointRenamingWith = {"T0": "0-96h", "T1": "0-96h", "T2": "0-96h", "T3": "0-96h", "T4": "0-96h"},
    #                                               )
    # mainPrintMultipleRelativeDiffBetweenScenarios(measureName="angleGiniCoeff", measuresBaseName="combinedMeasures.csv",
    #                                               listOfScenarioIdFrom=[('SAM', 'WT', '0-96h')],
    #                                               listOfScenarioIdTo=[('inflorescence meristem', 'ktn', '0-96h')],
    #                                               overwriteTimePointRenamingWith={"T0": "0-96h", "T1": "0-96h", "T2": "0-96h", "T3": "0-96h", "T4": "0-96h"}
    #
    #                                               )
    # printGeneralGDifferencesToIdentifyDirectionality()
    # print("only WT SAM")
    # mainPrintMeanValueOverScenarios(measureName="lengthGiniCoeff", tissueScenarioNames=["SAM"],
    #                                 furtherRemoveScenarioIds=[('inflorescence meristem', 'ktn', '0-96h')],
    #                                 overwriteTimePointRenamingWith={"T0": "0-96h", "T1": "0-96h", "T2": "0-96h", "T3": "0-96h", "T4":"0-96h"})
    # mainPrintMeanValueOverScenarios(measureName="angleGiniCoeff", tissueScenarioNames=["SAM"],
    #                                 furtherRemoveScenarioIds=[('inflorescence meristem', 'ktn', '0-96h')],
    #                                 overwriteTimePointRenamingWith={"T0": "0-96h", "T1": "0-96h", "T2": "0-96h", "T3": "0-96h", "T4":"0-96h"})
    # print("only ktn SAM")
    # mainPrintMeanValueOverScenarios(measureName="lengthGiniCoeff", tissueScenarioNames=["SAM"],
    #                                 furtherRemoveScenarioIds=[('SAM', 'WT', '0-96h')],
    #                                 overwriteTimePointRenamingWith={"T0": "0-96h", "T1": "0-96h", "T2": "0-96h", "T3": "0-96h", "T4":"0-96h"})
    # mainPrintMeanValueOverScenarios(measureName="angleGiniCoeff", tissueScenarioNames=["SAM"],
    #                                 furtherRemoveScenarioIds=[('SAM', 'WT', '0-96h')],
    #                                 overwriteTimePointRenamingWith={"T0": "0-96h", "T1": "0-96h", "T2": "0-96h", "T3": "0-96h", "T4":"0-96h"})
