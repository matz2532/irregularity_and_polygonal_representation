import itertools
import numpy as np
import pandas as pd
import scipy.stats as st
import sys

sys.path.insert(0, "./Code/Analyze/cldToSignificanceLetter/")

from cld import calcGroupLetters
from pathlib import Path
from scipy.stats import norm, shapiro, ttest_1samp, ttest_ind, wilcoxon
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd

class AnovaLetterPlotter (object):

    insideGroupingLetters=None
    # columns for accessing tissue name, gene name, and time point
    geneNameColName="gene name"
    timePointColName="time point numeric"
    tissueColName="tissue name"
    # p-value related parameters
    pValueThreshold=0.05
    performAnova=True
    pairwiseTestingDf=None
    vsTestingDf=None
    fontSize=14

    def __init__(self, fontSize=None, defaultFontSizeOffset=-8):
        if not fontSize is None:
            self.fontSize = fontSize + defaultFontSizeOffset

    def AddAllTestsOfTissueAndGenes(self, ax, table, entriyIdentifier, yName, testAllAgainstAllAnova=False,
                                    yOffsetFactor=0.03, yOffset=None, xOffset=0, xOffsets=None, additionalDistanceFactorPerTest=0.04,
                                    testVsValue=None, givenXTicks=None, capitalize=False, italicize=False):
        self.insideGroupingLetters = None
        self.pairwiseTestingDf = []
        self.pairwiseTukeyTestDf = None
        if not testVsValue is None:
            self.AddTTestAgainstValue(ax, testVsValue, table, yName, entriyIdentifier, yOffsetFactor=yOffsetFactor, yOffset=yOffset, xOffset=xOffset, xOffsets=xOffsets,
                                      useInsideGroupsMaxYPosition= not testAllAgainstAllAnova, givenXTicks=givenXTicks)
            yOffsetFactor += additionalDistanceFactorPerTest
        else:
            self.groupingLettersVsValue, self.vsTestingDf = None, None
        nrOfIdentifiers = len(entriyIdentifier)
        if nrOfIdentifiers > 1:
            if testAllAgainstAllAnova:
                self.AddAnovaMultiGroupLetters(ax, table, entriyIdentifier, yName,
                                               yOffsetFactor=yOffsetFactor, yOffset=yOffset, xOffset=xOffset, xOffsets=xOffsets, givenXTicks=givenXTicks, capitalize=capitalize, italicize=italicize)
                self.pairwiseTestingDf = None
            else:
                self.AddTTestInsideAndBetweenTissueAndGenes(ax, table, entriyIdentifier, yName,
                                                            yOffsetFactor=yOffsetFactor, yOffset=yOffset, xOffset=xOffset, xOffsets=xOffsets, capitalize=capitalize, italicize=italicize,
                                                            additionalDistanceFactorPerTest=additionalDistanceFactorPerTest, givenXTicks=givenXTicks)
        else:
            self.insideGroupingLetters, self.betweenGroupingLetters = "", ""
            self.pairwiseTestingDf = None

    def AddAnovaMultiGroupLetters(self, ax, table, entriyIdentifier, yName, yOffsetFactor=0.03, yOffset=None, xOffset=0, xOffsets=None, givenXTicks=None, capitalize=False, italicize=False):
        valuesOfAllIdentifiers = [self.extractColumnValuesOf(table, geneName, tissueName, numericTimePoint, yName ) for geneName, tissueName, numericTimePoint in entriyIdentifier]
        groupNames = [self.combineIdentifiers(i) for i in entriyIdentifier]
        i = 0
        toRemove = []
        for name, values in zip(groupNames, valuesOfAllIdentifiers):
            if len(values) < 2:
                print(f"The group {name} has less than 2 {values=} and is therefore discarded.")
                toRemove.append(i)
            i += 1
        groupNames = np.delete(groupNames, toRemove)
        valuesOfAllIdentifiers = np.delete(valuesOfAllIdentifiers, toRemove)
        assert len(valuesOfAllIdentifiers) == len(groupNames), f"Not each group has values {len(valuesOfAllIdentifiers)} != {len(groupNames)}\n{groupNames=}\n{valuesOfAllIdentifiers=}"
        self.tukeyTestText, rawPairwiseTukeyTest = self.doTukey(valuesOfAllIdentifiers, groupNames)
        self.pairwiseTukeyTestDf = self.convertPairwiseTukeyTestToDf(rawPairwiseTukeyTest, groupNames)
        self.anovaAllVsAllLetters = self.calculateGroupLettersFrom(self.pairwiseTukeyTestDf, entriyIdentifiersToOrderAlong=entriyIdentifier, capitalize=capitalize, italicize=italicize)
        if not ax is None:
            self.plotGroupNameLetters(ax, self.anovaAllVsAllLetters, entriyIdentifier, table, yName,
                                      yOffsetFactor=yOffsetFactor, yOffset=yOffset, xOffset=xOffset, xOffsets=xOffsets,
                                      useInsideGroupsMaxYPosition=False, givenXTicks=givenXTicks)

    def AddTTestInsideAndBetweenTissueAndGenes(self, ax, table, entriyIdentifier, yName, doInsideTesting=False,
                                               yOffsetFactor=0.03, yOffset=None, xOffset=0, xOffsets=None,
                                               additionalDistanceFactorPerTest=0.04, givenXTicks=None, capitalize=False, italicize=False):
        timePointGroupingOfTissueGeneId = self.determineInsideTissueAndGenesGroupings(entriyIdentifier)
        if doInsideTesting:
            testResultsInsideOfGroupings = self.testInsideGroupings(table, timePointGroupingOfTissueGeneId, yName, performAnova=self.performAnova)
        testResultsBetweenGroupings = self.testBetweenGroupings(table, timePointGroupingOfTissueGeneId, yName, performAnova=self.performAnova)
        if not self.performAnova:
            testResultsInsideOfGroupings, testResultsBetweenGroupings = self.multipleTestCorrect(testResultsInsideOfGroupings, testResultsBetweenGroupings)
            testResultsBetweenGroupings = self.addMinMaxOfInsideTestingTo(testResultsBetweenGroupings, testResultsInsideOfGroupings, timePointGroupingOfTissueGeneId)
        if doInsideTesting:
            self.insideGroupingLetters = self.calculateGroupingLettersOfArrayDict(testResultsInsideOfGroupings, entriyIdentifier, performAnova=self.performAnova, capitalize=capitalize, italicize=italicize)
            self.plotGroupNameLetters(ax, self.insideGroupingLetters, entriyIdentifier, table, yName, yOffsetFactor=yOffsetFactor, yOffset=yOffset, xOffset=xOffset, xOffsets=xOffsets, givenXTicks=givenXTicks)
            yOffsetFactor += additionalDistanceFactorPerTest
        self.betweenGroupingLetters = self.calculateGroupLettersOfArray(testResultsBetweenGroupings, entriyIdentifier, groupType="between groups", capitalize=not capitalize, italicize=italicize, performAnova = self.performAnova)
        self.plotGroupNameLetters(ax, self.betweenGroupingLetters, entriyIdentifier, table, yName, yOffsetFactor=yOffsetFactor, yOffset=yOffset, xOffset=xOffset, xOffsets=xOffsets, givenXTicks=givenXTicks)

    def AddTTestAgainstValue(self, ax, testVsValue, table, yName, entriyIdentifier, yOffsetFactor=0.03, yOffset=None, xOffset=0, xOffsets=None,
                             useInsideGroupsMaxYPosition=True, givenXTicks=None):
        self.groupingLettersVsValue = self.calculateGroupingLettersVsValue(testVsValue, table, yName, entriyIdentifier)
        self.plotGroupNameLetters(ax, self.groupingLettersVsValue, entriyIdentifier, table, yName, yOffsetFactor=yOffsetFactor, yOffset=yOffset, xOffset=xOffset, xOffsets=xOffsets, useInsideGroupsMaxYPosition=useInsideGroupsMaxYPosition, givenXTicks=givenXTicks)

    def SaveStatistic(self, baseFilename="", filenameExtension="_detailedStatistics.csv", filenameExtensionForVsValue="_vsValue.csv",
                      filenameExtensionForAllvsAllAnova="AllVsAllAnova.csv", exactFilenameToSave=None, sep=","):
        if exactFilenameToSave is None:
            exactFilenameToSave = self.replaceExtension(baseFilename, filenameExtension)
        Path(exactFilenameToSave).parent.mkdir(parents=True, exist_ok=True)
        savedFilenames = []
        if not self.pairwiseTestingDf is None:
            self.pairwiseTestingDf = pd.concat(self.pairwiseTestingDf)
            self.pairwiseTestingDf.to_csv(exactFilenameToSave, sep=sep, index=False)
            with open(exactFilenameToSave, "a") as fh:
                fh.write(f"\n")
                if not self.insideGroupingLetters is None:
                    fh.write(f"{self.insideGroupingLetters=}\n")
                fh.write(f"{self.betweenGroupingLetters=}\n")
            savedFilenames.append(exactFilenameToSave)
        if not self.vsTestingDf is None:
            adaptedFilenameToSave = self.replaceExtension(exactFilenameToSave, filenameExtensionForVsValue)
            Path(adaptedFilenameToSave).parent.mkdir(parents=True, exist_ok=True)
            self.vsTestingDf.to_csv(adaptedFilenameToSave, sep=sep, index=False)
            with open(adaptedFilenameToSave, "a") as fh:
                fh.write(f"\n")
                fh.write(f"{self.groupingLettersVsValue=}\n")
            savedFilenames.append(adaptedFilenameToSave)
        if not self.pairwiseTukeyTestDf is None:
            adaptedFilenameToSave = self.replaceExtension(exactFilenameToSave, filenameExtensionForAllvsAllAnova)
            Path(adaptedFilenameToSave).parent.mkdir(parents=True, exist_ok=True)
            self.pairwiseTukeyTestDf.to_csv(adaptedFilenameToSave, sep=sep, index=False)
            with open(adaptedFilenameToSave, "a") as fh:
                fh.write(f"\n")
                fh.write(f"{self.anovaAllVsAllLetters=}\n")
            savedFilenames.append(adaptedFilenameToSave)
        return savedFilenames

    def replaceExtension(self, filename, extensionReplacement, asString=False):
        newFilename = Path(filename).parent.joinpath(Path(filename).stem + extensionReplacement)
        if asString:
            newFilename= str(newFilename)
        return newFilename

    def doTukey(self, valuesOfGroup, name, alpha=0.05):
        if len(valuesOfGroup) >= 2:
            archive = dict(zip(name, valuesOfGroup))
            fvalue, pvalue = st.f_oneway(*archive.values())
        else:
            print("tukey HSD test with only on group is not implemented, {} < 2".format(len(valuesOfGroup)))
            sys.exit()
        groupName = [[g]*len(v) for g, v in zip(name, valuesOfGroup)]
        groupName = np.concatenate(groupName)
        valuesOfGroup = np.concatenate(valuesOfGroup)
        pairwiseTukeyTest = pairwise_tukeyhsd(endog=valuesOfGroup, groups=groupName, alpha=alpha)
        textResult = f"Results of ANOVA test:\n The F-statistic is: {fvalue}\n The p-value is: {pvalue}\n"+str(pairwiseTukeyTest)
        return textResult, pairwiseTukeyTest

    def convertPairwiseTukeyTestToDf(self, rawPairwiseTukeyTest, groupNames):
        group1 = []
        group2 = []
        reject = rawPairwiseTukeyTest.reject
        confidenceInterval = rawPairwiseTukeyTest.confint
        adjustedPValues = rawPairwiseTukeyTest.pvalues
        meandiffs = rawPairwiseTukeyTest.meandiffs
        for i, j in itertools.combinations(rawPairwiseTukeyTest.groupsunique, 2):
            group1.append(i)
            group2.append(j)
        pairwiseTukeyTestDf = pd.DataFrame({"group1": group1, "group2": group2, "reject": reject,
                                            "adjusted p-value" : adjustedPValues, "mean difference" : meandiffs,
                                            "lower conf. interval" : confidenceInterval[:, 0],
                                            "upper conf. interval" : confidenceInterval[:, 1]})
        groupNameMapping = {groupName:i for i, groupName in enumerate(groupNames)}
        pairwiseTukeyTestDf = (pairwiseTukeyTestDf.assign(key1=pairwiseTukeyTestDf.group1.map(groupNameMapping),
              key2=pairwiseTukeyTestDf.group2.map(groupNameMapping))
                .sort_values(['key1', 'key2'])
                .drop(['key1', 'key2'], axis=1))
        pairwiseTukeyTestDf.index = np.arange(len(pairwiseTukeyTestDf))
        return pairwiseTukeyTestDf

    def determineInsideTissueAndGenesGroupings(self, entriyIdentifier):
        timePointGroupingOfTissueGeneId = {}
        for currentIdentifier in entriyIdentifier:
            assert len(currentIdentifier) == 3, f"The current identifier {currentIdentifier=} does not have three entries (the tissue name, gene name, and time point) {len(currentIdentifier)} != 3"
            tissueName, geneName, timePoint = currentIdentifier
            tissueAndGeneIdentifier = (tissueName, geneName)
            if not tissueAndGeneIdentifier in timePointGroupingOfTissueGeneId:
                timePointGroupingOfTissueGeneId[tissueAndGeneIdentifier] = [timePoint]
            else:
                timePointGroupingOfTissueGeneId[tissueAndGeneIdentifier].append(timePoint)
        return timePointGroupingOfTissueGeneId

    def testInsideGroupings(self, table, timePointGroupingOfTissueGeneId, yName, performAnova=False):
        statisticResultsPerGroup = {}
        for currentIdentifier, insideIdentifiers in timePointGroupingOfTissueGeneId.items():
            if len(insideIdentifiers) < 2:
                continue
            assert len(currentIdentifier) == 2, f"The current identifier {currentIdentifier=} does not have three entries (the tissue, and gene name) {len(currentIdentifier)} != 2"
            tissueName, geneName = currentIdentifier
            if performAnova:
                valuesOfGroups, groupNames = [], []
                for timePoint in insideIdentifiers:
                    groupNames.append(timePoint)
                    valuesOfGroups.append(self.extractColumnValuesOf(table, tissueName, geneName, timePoint, yName))
                tukeyTestText, rawPairwiseTukeyTest = self.doTukey(valuesOfGroups, groupNames)
                pairwiseTukeyTestDf = self.convertPairwiseTukeyTestToDf(rawPairwiseTukeyTest, groupNames)
                statisticInsideOfGroup = []
                for idx in pairwiseTukeyTestDf.index:
                    insideGroup1 = pairwiseTukeyTestDf["group1"][idx]
                    insideGroup2 = pairwiseTukeyTestDf["group2"][idx]
                    adjustedPValue = pairwiseTukeyTestDf["adjusted p-value"][idx]
                    meanDifference = pairwiseTukeyTestDf["mean difference"][idx]
                    lowerConfInterval = pairwiseTukeyTestDf["lower conf. interval"][idx]
                    upperConfInterval = pairwiseTukeyTestDf["upper conf. interval"][idx]
                    statisticInsideOfGroup.append([tissueName, geneName, insideGroup1, insideGroup2, meanDifference, lowerConfInterval, upperConfInterval, adjustedPValue])  # in t-test last for last value is statistics instead of mean difference
                statisticResultsPerGroup[currentIdentifier] = statisticInsideOfGroup
            else:
                # perform t-test inside groups (between time points)
                statisticInsideOfGroup = []
                for timePoint1, timePoint2 in itertools.combinations(insideIdentifiers, 2):
                    firstValues = self.extractColumnValuesOf(table, tissueName, geneName, timePoint1, yName)
                    secondValues = self.extractColumnValuesOf(table, tissueName, geneName, timePoint2, yName)
                    tTestResults = ttest_ind(firstValues, secondValues)
                    statisticInsideOfGroup.append([tissueName, geneName, timePoint1, timePoint2, *tTestResults])
                statisticResultsPerGroup[currentIdentifier] = statisticInsideOfGroup
        return statisticResultsPerGroup

    def extractColumnValuesOf(self, table, tissueName, geneName, timePoint, yName):
        isTissueName = table[self.tissueColName] == tissueName
        isGeneName = table[self.geneNameColName] == geneName
        isTimePoint = table[self.timePointColName] == timePoint
        isSelected = isTissueName & isGeneName & isTimePoint
        assert np.sum(isSelected) > 0, f"The samples of {tissueName=} {geneName=} {timePoint=} have no entries.\nThere are {np.sum(isTissueName)} values of this type present in tissue name column {self.tissueColName} only {table[self.tissueColName].unique()} are present.\nThere are {np.sum(isGeneName)} values of this type present in gene name column {self.geneNameColName} only {table[self.geneNameColName].unique()} are present.\nThere are {np.sum(isTimePoint)} values of this type present in time point column {self.timePointColName} only {table[self.timePointColName].unique()} are present."
        return table[isSelected][yName].to_numpy(dtype=float)

    def testBetweenGroupings(self, table, timePointGroupingOfTissueGeneId, yName, performAnova=False):
        if self.performAnova:
            betweenGroupsTestResults = self.testBetweenGroupsWithAnova(table, timePointGroupingOfTissueGeneId, yName)
        else:
            betweenGroupsTestResults = self.testBetweenGroupsWithTTest(table, timePointGroupingOfTissueGeneId, yName)
        return betweenGroupsTestResults

    def testBetweenGroupsWithAnova(self, table, timePointGroupingOfTissueGeneId, yName):
        valuesOfGroups, groupNames, = [], []
        groupNamesToGroupListConverter = {}
        groupIdentifiers = list(timePointGroupingOfTissueGeneId.keys())
        for tissueTypeAndGeneName in groupIdentifiers:
            existingTimePoints = timePointGroupingOfTissueGeneId[tissueTypeAndGeneName]
            assert len(existingTimePoints) > 0, f"The given timePointGroupingOfTissueGeneId does not contain any time points. {timePointGroupingOfTissueGeneId=}"
            minTimePoint = np.min(existingTimePoints)
            maxTimePoint = np.max(existingTimePoints)
            currentGroupName = self.combineIdentifiers([*tissueTypeAndGeneName, minTimePoint])
            groupNames.append(currentGroupName)
            groupNamesToGroupListConverter[currentGroupName] = [*tissueTypeAndGeneName, minTimePoint]
            valuesOfCurrentGroup = self.extractColumnValuesOf(table, *tissueTypeAndGeneName, minTimePoint, yName)
            valuesOfGroups.append(valuesOfCurrentGroup)
            if minTimePoint != maxTimePoint:
                currentGroupName = self.combineIdentifiers([*tissueTypeAndGeneName, maxTimePoint])
                groupNames.append(currentGroupName)
                groupNamesToGroupListConverter[currentGroupName] = [*tissueTypeAndGeneName, maxTimePoint]
                valuesOfCurrentGroup = self.extractColumnValuesOf(table, *tissueTypeAndGeneName, maxTimePoint, yName)
                valuesOfGroups.append(valuesOfCurrentGroup)
        tukeyTestText, rawPairwiseTukeyTest = self.doTukey(valuesOfGroups, groupNames)
        pairwiseTukeyTestDf = self.convertPairwiseTukeyTestToDf(rawPairwiseTukeyTest, groupNames)
        # not best approach, but integrates best with existing code :(
        betweenGroupsTestResults = []
        for idx in pairwiseTukeyTestDf.index:
            group1 = pairwiseTukeyTestDf["group1"][idx]
            group2 = pairwiseTukeyTestDf["group2"][idx]
            adjustedPValue = pairwiseTukeyTestDf["adjusted p-value"][idx]
            meanDifference = pairwiseTukeyTestDf["mean difference"][idx]
            lowerConfInterval = pairwiseTukeyTestDf["lower conf. interval"][idx]
            upperConfInterval = pairwiseTukeyTestDf["upper conf. interval"][idx]
            groupIdentifier1 = groupNamesToGroupListConverter[group1]
            groupIdentifier2 = groupNamesToGroupListConverter[group2]
            betweenGroupsTestResults.append([*groupIdentifier1, *groupIdentifier2, meanDifference, lowerConfInterval, upperConfInterval, adjustedPValue])  # in t-test last for last value is statistics instead of mean difference
        return betweenGroupsTestResults

    def testBetweenGroupsWithTTest(self, table, timePointGroupingOfTissueGeneId, yName):
        groupIdentifiers = list(timePointGroupingOfTissueGeneId.keys())
        tTestResultsBetweenGroups = []
        for groupIdentifier1, groupIdentifier2 in itertools.combinations(groupIdentifiers, 2):
            timePointsOf1 = timePointGroupingOfTissueGeneId[groupIdentifier1]
            timePointsOf2 = timePointGroupingOfTissueGeneId[groupIdentifier2]
            minTimePoint1 = np.min(timePointsOf1)
            minTimePoint2 = np.min(timePointsOf2)
            minFirstValues = self.extractColumnValuesOf(table, *groupIdentifier1, minTimePoint1, yName)
            minSecondValues = self.extractColumnValuesOf(table, *groupIdentifier2, minTimePoint2, yName)
            minMinTTestResults = ttest_ind(minFirstValues, minSecondValues)
            tTestResultsBetweenGroups.append([*groupIdentifier1, minTimePoint1, *groupIdentifier2, minTimePoint2, *minMinTTestResults])
            if len(timePointsOf1) > 1:
                maxTimePoint1 = np.max(timePointsOf1)
                maxFirstValues = self.extractColumnValuesOf(table, *groupIdentifier1, maxTimePoint1, yName)
                maxMinTTestResults = ttest_ind(maxFirstValues, minSecondValues)
                tTestResultsBetweenGroups.append([*groupIdentifier1, maxTimePoint1, *groupIdentifier2, minTimePoint2, *maxMinTTestResults])
            if len(timePointsOf2) > 1:
                maxTimePoint2 = np.max(timePointsOf2)
                maxSecondValues = self.extractColumnValuesOf(table, *groupIdentifier2, maxTimePoint2, yName)
                minMaxTTestResults = ttest_ind(minFirstValues, maxSecondValues)
                tTestResultsBetweenGroups.append([*groupIdentifier1, minTimePoint1, *groupIdentifier2, maxTimePoint2, *minMaxTTestResults])
            if len(timePointsOf1) > 1 and len(timePointsOf2) > 1:
                maxMaxTTestResults = ttest_ind(maxFirstValues, maxSecondValues)
                tTestResultsBetweenGroups.append([*groupIdentifier1, maxTimePoint1, *groupIdentifier2, maxTimePoint2, *maxMaxTTestResults])
        return tTestResultsBetweenGroups

    def multipleTestCorrect(self, testResults1, testResults2):
        allPValues = []
        for i in testResults1.values():
            for j in i:
                allPValues.append(j[-1])
        allPValues.extend([i[-1] for i in testResults2])
        if len(allPValues) > 1:
            correctedPValues = multipletests(allPValues, method='fdr_bh')[1]
        else:
            correctedPValues = allPValues
        i = 0
        for result in testResults1.values():
            for pairedResults in result:
                pairedResults.append(correctedPValues[i])
                i += 1
        for j, k in enumerate(range(i, i+len(testResults2))):
            testResults2[j].append(correctedPValues[k])
        return testResults1, testResults2

    def addMinMaxOfInsideTestingTo(self, testResultsBetweenGroupings, testResultsInsideOfGroupings, timePointGroupingOfTissueGeneId):
        for insideGroupIdentifier, insideGroupResults in testResultsInsideOfGroupings.items():
            timePointsOf = timePointGroupingOfTissueGeneId[insideGroupIdentifier]
            if len(timePointsOf) > 1:
                minTimePoint = np.min(timePointsOf)
                maxTimePoint = np.max(timePointsOf)
                insideGroupArray = np.array(insideGroupResults, dtype=object)
                isMinOrMax = np.isin(insideGroupArray[:, [2, 3]], [minTimePoint, maxTimePoint])
                isRowMinMaxComparison = np.sum(isMinOrMax, axis=1) == 2
                if self.performAnova:
                    pValueResults = insideGroupArray[isRowMinMaxComparison, [4, 5, 6, 7]]
                else:
                    pValueResults = insideGroupArray[isRowMinMaxComparison, [4, 5, 6]]
                testResultsBetweenGroupings.append([*insideGroupIdentifier, minTimePoint, *insideGroupIdentifier, maxTimePoint, *pValueResults])
        return testResultsBetweenGroupings

    def calculateGroupingLettersOfArrayDict(self, testResultsInsideOfGroupings, entriyIdentifiersToOrderAlong, capitalize=False, italicize=False, performAnova=False):
        insideGroupNamesLetters = {}
        for tissueAndGeneIdentifier, resultsTable in testResultsInsideOfGroupings.items():
            groupType = f"in group_{tissueAndGeneIdentifier}"
            groupNamesLetters = self.calculateGroupLettersOfArray(resultsTable, entriyIdentifiersToOrderAlong, groupType, secondIdentifierIdx=[0, 1, 3],
                                                                  capitalize=capitalize, italicize=italicize, performAnova=performAnova)
            for groupName, letters in groupNamesLetters.items():
                insideGroupNamesLetters[groupName] = letters
        return insideGroupNamesLetters

    def calculateGroupLettersOfArray(self, testResultsBetweenGroupings, entriyIdentifiersToOrderAlong, groupType="",
                                     firstIdentifierIdx=[0, 1, 2], secondIdentifierIdx=[3, 4, 5],
                                     capitalize=False, italicize=False, performAnova=False):
        testResults = np.array(testResultsBetweenGroupings, dtype=object)
        group1 = [self.combineIdentifiers(identifier) for identifier in testResults[:, firstIdentifierIdx]]
        group2 = [self.combineIdentifiers(identifier) for identifier in testResults[:, secondIdentifierIdx]]
        pValues = testResults[:, -1].astype(float)
        pairwiseTestingDf = {"test type":len(pValues)*[groupType], "group1":group1, "group2":group2}
        if performAnova:
            meanDifferences = testResults[:, -4]
            lowerConfInterval = testResults[:, -3]
            upperConfInterval = testResults[:, -2]
            pairwiseTestingDf["mean difference"] = meanDifferences
            pairwiseTestingDf["lower conf. interval"] = lowerConfInterval
            pairwiseTestingDf["upper conf. interval"] = upperConfInterval
        else:
            unadjustedPValues = testResults[:, -2].astype(float)
            pairwiseTestingDf["unadjusted p-value"] = unadjustedPValues
        pairwiseTestingDf["reject"] = pValues < self.pValueThreshold
        pairwiseTestingDf["adjusted p-value"] = pValues
        pairwiseTestingDf = pd.DataFrame(pairwiseTestingDf)
        groupNamesLetters = self.calculateGroupLettersFrom(pairwiseTestingDf, entriyIdentifiersToOrderAlong, capitalize=capitalize, italicize=italicize)
        self.pairwiseTestingDf.append(pairwiseTestingDf)
        return groupNamesLetters

    def calculateGroupLettersFrom(self, pairwiseTestingDf, entriyIdentifiersToOrderAlong=None, capitalize=False, italicize=False, groupNames=None):
        if groupNames is None:
            groupNames = [identifier if type(identifier) == str else self.combineIdentifiers(identifier) for identifier in entriyIdentifiersToOrderAlong]
        groupNamesLetters = calcGroupLetters(pairwiseTestingDf, col1="group1", col2="group2",
                                             rejectCol="reject", orderGroupsAlong=groupNames)
        groupNamesLetters = self.correctGroupLettersOccurence(groupNamesLetters)
        groupNamesLetters = self.alphabeticallyOrderDictValues(groupNamesLetters)
        groupNamesLetters = self.limitPerLineDictValuesLength(groupNamesLetters)
        if capitalize:
            groupNamesLetters = self.capitalizeGroupLetters(groupNamesLetters)
        if italicize:
            groupNamesLetters = self.italicizeGroupLetters(groupNamesLetters)
        return groupNamesLetters

    def combineIdentifiers(self, identifiers, sep="_"):
        return sep.join([str(i) for i in identifiers])

    def correctGroupLettersOccurence(self, groupNamesLetters):
        allLetterAsInts = []
        for letters in groupNamesLetters.values():
            for l in letters:
                letterAsInt = self.changeLetterToInt(l)
                allLetterAsInts.append(letterAsInt)
        allLetterAsInts = np.unique(allLetterAsInts)
        changedLetterAsInts = np.array(allLetterAsInts)
        differenceToPredecesor = np.diff(np.concatenate([[-1], allLetterAsInts]))
        isDiffBiggerThanOne = differenceToPredecesor > 1
        if np.any(isDiffBiggerThanOne):
            indicesOfChanges = np.where(isDiffBiggerThanOne)[0]
            for i in indicesOfChanges:
                changedLetterAsInts[i:] -= differenceToPredecesor[i] - 1
            allLettersToConvert = {}
            for i in np.arange(np.min(indicesOfChanges), len(changedLetterAsInts)):
                letter = self.changeIntToLetter(allLetterAsInts[i])
                changeToLetter = self.changeIntToLetter(changedLetterAsInts[i])
                allLettersToConvert[letter] = changeToLetter
            for k, letters in groupNamesLetters.items():
                for changeLetterFrom, toLetter in allLettersToConvert.items():
                    letters = letters.replace(changeLetterFrom, toLetter)
                groupNamesLetters[k] = letters
        return groupNamesLetters

    def changeLetterToInt(self, letter):
        letterAsInt = ord(letter)
        if letterAsInt > 96:
            letterAsInt -= 97
        else:
            letterAsInt -= 39
        return letterAsInt

    def changeIntToLetter(self, letterAsInt):
        if letterAsInt > 25:
            letter = chr(letterAsInt + 39)
        else:
            letter = chr(letterAsInt + 97)
        return letter

    def alphabeticallyOrderDictValues(self, stringDict, delim=""):
        for key, values in stringDict.items():
            alphabeticallyOrderedValues = delim.join(sorted(values))
            stringDict[key] = alphabeticallyOrderedValues
        return stringDict

    def limitPerLineDictValuesLength(self, stringDict, symbolsPerLine=3, newLineDelim="\n"):
        for key, values in stringDict.items():
            perLineSplitString = [values[i:i+symbolsPerLine] for i in range(len(values))[::symbolsPerLine]]
            alphabeticallyOrderedValues = newLineDelim.join(perLineSplitString)
            stringDict[key] = alphabeticallyOrderedValues
        return stringDict

    def capitalizeGroupLetters(self, groupNamesLetters):
        for groupName, letters in groupNamesLetters.items():
            groupNamesLetters[groupName] = letters.upper()
        return groupNamesLetters

    def italicizeGroupLetters(self, groupNamesLetters):
        for groupName, letters in groupNamesLetters.items():
            groupNamesLetters[groupName] = "$\it{" + letters + "}$"
        return groupNamesLetters

    def calculateGroupingLettersVsValue(self, testVsValue, table, yName, entriyIdentifier, normalityAlpha=0.05, alwaysDoTTest=True):
        vsTypeName = f"one sample t-test vs {testVsValue}"
        columnNames = ["test type", "tissue name", "gene name", "time point", "t-statistic", "unadjusted p-value"]
        vsTestingDf = []
        for currentIdentifier in entriyIdentifier:
             assert len(currentIdentifier) == 3, f"The current identifier {currentIdentifier=} does not have three entries (the tissue name, gene name, and time point) {len(currentIdentifier)} != 3"
             tissueName, geneName, timePoint = currentIdentifier
             yValues = self.extractColumnValuesOf(table, tissueName, geneName, timePoint, yName)
             statsOfShapiro, pValueOfShapiro = shapiro(yValues)
             if pValueOfShapiro < normalityAlpha and not alwaysDoTTest:
                 rank, pValue = wilcoxon(yValues-testVsValue, zero_method = "wilcox", correction=False)
                 tTest = norm.ppf(pValue/2)
             else:
                 tTest, pValue = ttest_1samp(yValues, popmean = testVsValue)
             vsTestingDf.append([vsTypeName, tissueName, geneName, timePoint, tTest, pValue])
        vsTestingDf = pd.DataFrame(vsTestingDf, columns = columnNames)
        unadjustedPValues = vsTestingDf["unadjusted p-value"]
        if len(unadjustedPValues) > 1:
            correctedPValues = list(multipletests(unadjustedPValues, method='fdr_bh'))[1]
        else:
            correctedPValues = unadjustedPValues
        vsTestingDf["corrected p-value"] = correctedPValues
        groupNamesLetters = self.convertSignificancesToGroupLetters(entriyIdentifier, correctedPValues)
        self.vsTestingDf = vsTestingDf
        return groupNamesLetters

    def convertSignificancesToGroupLetters(self, entriyIdentifier, allPValues):
        AllGroupNames = [self.combineIdentifiers(i) for i in entriyIdentifier]
        groupNamesLetters = {}
        for groupName, pValue in zip(AllGroupNames, allPValues):
            groupNamesLetters[groupName] = self.convertPValueToText(pValue)
        return groupNamesLetters

    def convertPValueToText(self, pValue, displayNonSignifance=True):
        if pValue >= 0.05:
            if displayNonSignifance:
                significanceText = "n.s."
            else:
                significanceText = ""
        elif pValue >= 0.01:
            significanceText = "*"
        elif pValue >= 0.001:
            significanceText = "**"
        else:
            significanceText = "***"
        return significanceText

    def plotGroupNameLetters(self, ax, groupNamesLetters, entriyIdentifier, table, yName,
                             yOffsetFactor=0, yOffset=None, xOffset=0,  xOffsets=None, increaseYOffsetFor="*", increaseYOffsetBy=-0.015,
                             useInsideGroupsMaxYPosition=False, givenXTicks=None):
        if givenXTicks is None:
            givenXTicks = np.arange(len(entriyIdentifier))
        groupNames = [self.combineIdentifiers(i) for i in entriyIdentifier]
        assert len(groupNames) == len(givenXTicks), f" The number of group names is not equal to the number of positions to add the letters of the group, {len(groupNames)} != {len(givenXTicks)}.\n {groupNames=}, {givenXTicks=} "
        xPosDict = dict(zip(groupNames, givenXTicks))
        maxValue = np.abs(table[yName].max())
        if useInsideGroupsMaxYPosition:
            yPosDict = self.determineInsideGroupsEntryYPosition(entriyIdentifier, table, yName)
        else:
            yPosDict = {identifierName : maxValue for identifierName in groupNames}
        for i, (identifierName, letters) in enumerate(groupNamesLetters.items()):
            if increaseYOffsetFor in letters:
                yOffsetFactor += increaseYOffsetBy
            if xOffsets is None:
                x = xPosDict[identifierName] + xOffset
            else:
                x = xOffsets[i]
            if yOffset is None:
                y = yPosDict[identifierName] + yOffsetFactor * maxValue
            else:
                y = yOffset + yOffsetFactor * yOffset
            if increaseYOffsetFor in letters:
                yOffsetFactor -= increaseYOffsetBy
            ax.text(x, y, letters, ha="center", size=self.fontSize, verticalalignment='center')

    def determineInsideGroupsEntryYPosition(self, entriyIdentifier, table, yName):
        groupedTables = table.groupby([self.tissueColName, self.geneNameColName], sort=False)
        tissueGeneYPosDict = {}
        for tissueGeneNameId, groupTable in groupedTables:
            tissueGeneYPosDict[tissueGeneNameId] = groupTable[yName].max()
        yPosDict = {}
        for currentIdentifier in entriyIdentifier:
            assert len(currentIdentifier) == 3, f"The current identifier {currentIdentifier=} does not have three entries (the tissue name, gene name, and time point) {len(currentIdentifier)} != 3"
            tissueName, geneName, timePoint = currentIdentifier
            groupName = self.combineIdentifiers(currentIdentifier)
            yPosDict[groupName] = tissueGeneYPosDict[(tissueName, geneName)]
        return yPosDict
