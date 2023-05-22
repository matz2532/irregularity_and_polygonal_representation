import colorsys
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import warnings

sys.path.insert(0, "./Code/DataStructures/")

from AnovaLetterPlotter import AnovaLetterPlotter
from CorrelationDataPlotter import CorrelationDataPlotter
from ResultsTableLoader import ResultsTableLoader
from pathlib import Path

class ExtendedPlotter(ResultsTableLoader):
    # default conversion dicts (provided key, returns 'converted' value)
    columnsToMergeDtypeDict = {"genotype": str, "replicate id": str, "time point": str}
    genotypeColorConversion = None
    labelNameConverterDict = {"lengthGiniCoeff": "Gini coefficient of length",
                              "angleGiniCoeff": "Gini coefficient of angle",
                              "lengthGiniCoeff_ignoringGuardCells": "Gini coefficient of length", "angleGiniCoeff_ignoringGuardCells": "Gini coefficient of angle",
                              "ratio_originalPolygonArea_labelledImageArea": "polygonal area ratio",
                              "ratio_regularPolygonArea_labelledImageArea": "regular polygonal area ratio"
                              }
    # default column names
    numericTimePointColName = "time point numeric"
    pureGenotypeColName = "gene name"
    replicateIdColName = "replicate id"
    timePointNameColName = "time point"
    tissueColName = "tissue name"
    # default initialized parameters
    combinedColumnName = None
    combinedColumnTypeName = None
    resultsTable = None

    def __init__(self, resultsTableFilenames, timePointConversionFromStrToNumber=None, renameGenotypesDict=None,
                 renameTimePointDict=None, onlyShowLastAndFirstTimePoint=False, furtherRemoveScenarioIds=None, reduceResultsToOnePerTissue=False):
        super().__init__(resultsTableFilenames, timePointConversionFromStrToNumber=timePointConversionFromStrToNumber, renameGenotypesDict=renameGenotypesDict,
                         renameTimePointDict=renameTimePointDict, onlyShowLastAndFirstTimePoint=onlyShowLastAndFirstTimePoint,
                         furtherRemoveScenarioIds=furtherRemoveScenarioIds, reduceResultsToOnePerTissue=reduceResultsToOnePerTissue)

    def CreateBoxPlot(self, measureCol, tissueGeneNameOrdering, x="orderedTimePoints", resultsTable=None, filenameToSave=None, savePlot=True,
                      genotypeColorConversion=None, increaseBounds=True, renameTimePointDict=None,
                      testVsValue=None, figsize=(25, 12), fontSize=35, nextAxisYDistance=-0.075, ylim=None, ylimMin=None, yLabePad=25, xPos=-0.05,
                      addLinearRegression=False, doStatistics=True, compareAllAgainstAll=False, fixxedYLocator=None, addHLineAt=None,
                      givenXTicks=None, showCombinedColumnsNextToEachOther=True, drawLinesAtTicksParameter="tissueTicks", showMinimalXLabels=False, excludeYLabel=True,
                      onlyPlotXLabel=None, useThisSingleColorInPlot=None, overwriteColors=True, logNormalize=False, capitalizeNumberOfSamples=False):
        plt.rcParams["font.size"] = fontSize
        plt.rcParams["font.family"] = "Arial"
        mpl.rcParams['axes.spines.right'] = False
        mpl.rcParams['axes.spines.top'] = False
        if resultsTable is None:
            resultsTable = self.resultsTable
        isCombinedTable = isinstance(resultsTable, list)
        if isCombinedTable:
            resultsTable = [self.removeNAs(table, measureCol) for table in resultsTable]
            axisParameter, resultsTable = self.addOrderedTimePointsOfResultsList(resultsTable, tissueGeneNameOrdering, overwriteColors=overwriteColors,
                                                                                 genotypeColorConversion=genotypeColorConversion, increaseBounds=increaseBounds,
                                                                                 showCombinedColumnsNextToEachOther=showCombinedColumnsNextToEachOther, useThisSingleColorInPlot=useThisSingleColorInPlot)
        else:
            resultsTable = self.removeNAs(resultsTable, measureCol)
            axisParameter, resultsTable = self.addOrderTimePoints(resultsTable, tissueGeneNameOrdering, genotypeColorConversion=genotypeColorConversion, increaseBounds=increaseBounds, useThisSingleColorInPlot=useThisSingleColorInPlot)
        if logNormalize:
            resultsTable[measureCol] = np.log2(resultsTable[measureCol])
        if not renameTimePointDict is None:
            timePointNames = np.asarray(axisParameter["timePointNames"])
            for replaceTimePoint, withTimePoint in renameTimePointDict.items():
                if replaceTimePoint in timePointNames:
                    timePointNames[replaceTimePoint == timePointNames] = withTimePoint
            axisParameter["timePointNames"] = timePointNames
        fig, ax1 = plt.subplots(num="TEST", figsize=figsize, constrained_layout=True)
        boxprops = dict(edgecolor="white")
        medianlineprops = dict(color="white")
        meanprops = dict(marker="o", markeredgecolor="black", markerfacecolor="white")
        palette = dict(zip(np.unique(resultsTable[x]), axisParameter["colorOfGenAtTimePoint"]))
        g = sns.boxplot(data=resultsTable, x=x, y=measureCol, ax=ax1, showmeans=True, boxprops=boxprops, medianprops=medianlineprops, meanprops=meanprops, palette=palette)
        self.drawVerticalLinesAt(axisParameter[drawLinesAtTicksParameter][1:-1])
        yOffsetFactor = 0.05
        if not genotypeColorConversion is None:
            colorOfGenAtTimePoint = axisParameter["colorOfGenAtTimePoint"]
            for i, artist in enumerate(g.artists):
                artist.set_facecolor(colorOfGenAtTimePoint[i])
        if doStatistics:
            if isCombinedTable and "xOffsets" in axisParameter:
                self.anovaPlotter = []
                yOffset = resultsTable[self.combinedColumnName].max()
                numberOfColumnTypes = len(resultsTable[self.combinedColumnTypeName].unique())
                xOffsets = np.array(axisParameter["xOffsets"])
                for i, (typeId, tableOfType) in enumerate(resultsTable.groupby([self.combinedColumnTypeName], sort=False)):
                    anovaPlotter = AnovaLetterPlotter(fontSize=fontSize)
                    capitalize = i % 2 == 1
                    italicize = i >= 2
                    if showCombinedColumnsNextToEachOther:
                        idx = np.arange(len(xOffsets))
                        idx = idx[np.mod(idx, numberOfColumnTypes) == i]
                        selectedXOffsets = xOffsets[idx]
                        entryIdentifier = [axisParameter["entryIdentifier"][j] for j in idx]
                    else:
                        selectedXOffsets = xOffsets[i * numberOfColumnTypes:((i + 1) * numberOfColumnTypes)]
                        entryIdentifier = axisParameter["entryIdentifier"][i * numberOfColumnTypes:((i + 1) * numberOfColumnTypes)]

                    anovaPlotter.AddAllTestsOfTissueAndGenes(ax1, tableOfType.reset_index(drop=True), entryIdentifier,
                                                             measureCol, testVsValue=testVsValue, testAllAgainstAllAnova=compareAllAgainstAll,
                                                             yOffset=yOffset, xOffsets=selectedXOffsets, yOffsetFactor=yOffsetFactor, givenXTicks=givenXTicks,
                                                             capitalize=capitalize, italicize=italicize)
                    self.anovaPlotter.append(anovaPlotter)
            else:
                self.anovaPlotter = AnovaLetterPlotter(fontSize=fontSize)
                self.anovaPlotter.AddAllTestsOfTissueAndGenes(ax1, resultsTable, axisParameter["entryIdentifier"],
                                                              measureCol, testVsValue=testVsValue, testAllAgainstAllAnova=compareAllAgainstAll,
                                                              yOffsetFactor=yOffsetFactor, givenXTicks=givenXTicks)
        if onlyPlotXLabel is None:
            self.plotAxis(ax1, axisParameter, nextAxisYDistance=nextAxisYDistance, xPos=xPos, showMinimalXLabels=showMinimalXLabels)
            ax1.set_xlabel("time point")
            if excludeYLabel:
                ax1.set_xlabel("")
        else:
            ax1.xaxis.set_major_formatter(ticker.FixedFormatter(axisParameter["allGeneNames"]))
            plt.xlabel(onlyPlotXLabel)
            if excludeYLabel:
                plt.xlabel("")

        if not addHLineAt is None:
            xMin = axisParameter[drawLinesAtTicksParameter][0]
            xMax = axisParameter[drawLinesAtTicksParameter][-1]
            ax1.hlines(addHLineAt, xMin, xMax, zorder=-1)

        if measureCol in self.labelNameConverterDict:
            measureName = self.labelNameConverterDict[measureCol]
        else:
            measureName = measureCol
        ax1.set_ylabel(measureName, labelpad=yLabePad)
        if not ylim is None:
            ax1.set_ylim(ylim)
        elif not ylimMin is None:
            ylimMax = ax1.get_ylim()[1]
            ax1.set_ylim((ylimMin, ylimMax))
        groupColumnsBy = [self.pureGenotypeColName, self.tissueColName, self.timePointNameColName]
        if addLinearRegression:
            butShowOnlyFor = self.pureGenotypeColName
        else:
            butShowOnlyFor = self.tissueColName
        textOfGroupValuePairsToAdd = self.determineNumberOfSamplesForValuesOfColumns(resultsTable, groupColumnsBy=groupColumnsBy, butShowOnlyFor=butShowOnlyFor, capitalize=capitalizeNumberOfSamples)
        if addLinearRegression:
            textOfGroupValuePairsToAdd, pooledRegressionResults = self.drawLinearRegression(resultsTable, yColName=measureCol, ax=ax1, axisParameter=axisParameter, textOfGroupValuePairsToAdd=textOfGroupValuePairsToAdd)
        if textOfGroupValuePairsToAdd:
            self.addTextOfGroup(textOfGroupValuePairsToAdd, resultsTable, x, measureCol, ax=ax1)
        if not fixxedYLocator is None:
            ax1.yaxis.set_major_locator(fixxedYLocator)

        if not givenXTicks is None:
            ax1.set_xticks(givenXTicks)
        if not filenameToSave is None:
            Path(filenameToSave).parent.mkdir(parents=True, exist_ok=True)
            if savePlot:
                plt.savefig(filenameToSave, bbox_inches="tight", dpi=300)
                plt.close()
            if doStatistics:
                if type(self.anovaPlotter) == list:
                    if self.combinedColumnTypeName in resultsTable.columns:
                        extensions = resultsTable[self.combinedColumnTypeName].unique()
                    else:
                        extensions = None
                    for i, anovaPlotter in enumerate(self.anovaPlotter):
                        if not extensions is None:
                            filenameExtension = "_{}_detailedStatistics.csv".format(extensions[i])
                        else:
                            filenameExtension = "_detailedStatistics.csv"
                        anovaPlotter.SaveStatistic(baseFilename=filenameToSave, filenameExtension=filenameExtension)
                else:
                    self.anovaPlotter.SaveStatistic(baseFilename=filenameToSave)
        else:
            plt.show()

    def removeNAs(self, resultsTable, columnToCheck):
        notNa = resultsTable[columnToCheck].notna()
        notNaIndices = np.where(notNa)[0]
        resultsTable = resultsTable.iloc[notNaIndices, :].reset_index(drop=True)
        return resultsTable

    def addOrderedTimePointsOfResultsList(self, resultsTable, tissueGeneNameOrdering, genotypeColorConversion=None, increaseBounds=False,
                                          showCombinedColumnsNextToEachOther=False, startLocation=0, overwriteColors=True, colorsOfColumnTypes=[(0, 114 / 255, 178 / 255), (213 / 255, 94 / 255, 0 / 255)], **kwargs):
        if not showCombinedColumnsNextToEachOther:
            pooledAxisParameter, resultsTable = self.determineAxisParametersOfCombinedColumnsAfterOneAnother(resultsTable, tissueGeneNameOrdering,
                                                                                               genotypeColorConversion=genotypeColorConversion, increaseBounds=increaseBounds,
                                                                                               startLocation=startLocation, **kwargs)
            colorOfGenAtTimePoint = [*(len(pooledAxisParameter["colorOfGenAtTimePoint"]) // 2) * [(0, 114 / 255, 178 / 255)], *(len(pooledAxisParameter["colorOfGenAtTimePoint"]) // 2) * [(213 / 255, 94 / 255, 0 / 255)]]
            resultsTable = pd.concat(resultsTable).reset_index(drop=True)
        else:
            nrOfCombinedColumns = len(resultsTable)
            colorPalette = sns.color_palette("colorblind")
            colorsOfColumnTypes = [colorPalette[i] for i in range(nrOfCombinedColumns)]
            colorOfGenAtTimePoint = len(resultsTable[0]) * colorsOfColumnTypes
            resultsTable = pd.concat(resultsTable).reset_index(drop=True)
            pooledAxisParameter, resultsTable = self.addOrderTimePoints(resultsTable, tissueGeneNameOrdering, genotypeColorConversion=genotypeColorConversion,
                                                          increaseBounds=increaseBounds, startLocation=startLocation,
                                                          sortDifferentColumnTypesNextToEachOther=True, **kwargs)
        if overwriteColors:
            pooledAxisParameter["colorOfGenAtTimePoint"] = colorOfGenAtTimePoint
        return pooledAxisParameter, resultsTable

    def determineAxisParametersOfCombinedColumnsAfterOneAnother(self, resultsTable, tissueGeneNameOrdering, genotypeColorConversion=None, increaseBounds=False,
                                                                startLocation=0, **kwargs):
        allAxisParameter = []
        allStartLocations = []
        allTablesAgain = []
        for i, table in enumerate(resultsTable):
            allStartLocations.append(startLocation)
            includeFirstStartLocation = i == 0
            axisParameter, table = self.addOrderTimePoints(table, tissueGeneNameOrdering, genotypeColorConversion=genotypeColorConversion,
                                                    increaseBounds=increaseBounds, startLocation=startLocation,
                                                    includeFirstStartLocation=includeFirstStartLocation, increaseStartBounds=includeFirstStartLocation, **kwargs)
            startLocation = np.max(table["orderedTimePoints"]) + 1
            if not includeFirstStartLocation:
                axisParameter["genTicks"].pop(0)
                axisParameter["tissueTicks"].pop(0)
            allAxisParameter.append(axisParameter)
            allTablesAgain.append(table)
        pooledAxisParameter = self.poolDictsByExtendingLists(allAxisParameter, keysToAppend=["entryIdentifier"])
        pooledAxisParameter["allStartLocations"] = allStartLocations
        return pooledAxisParameter, allTablesAgain

    def addOrderTimePoints(self, resultsTable, tissueGeneNameOrdering, genotypeColorConversion=None,
                           increaseBounds=False, increaseStartBounds=True, increaseEndBounds=True,
                           warnAboutTimeIdHavingMoreThanOneName=True, enforceTimePointIdHavingOneName=False,
                           startLocation=0, includeFirstStartLocation=True, sortDifferentColumnTypesNextToEachOther=False,
                           useThisSingleColorInPlot=None):
        resultsTable = self.removeTableEntriesNotPresentIn(resultsTable, tissueGeneNameOrdering)
        numberOfCells = len(resultsTable)
        orderedTimePoints = np.zeros(numberOfCells, dtype=int)
        timePointNames, entryIdentifier, xOffsets = [], [], []
        genTicks, geneLocations, allGeneNames = [], [], []
        tissueTicks, tissueLocations, allTissueNames = [], [], []
        if includeFirstStartLocation:
            genTicks.append(startLocation)
            tissueTicks.append(startLocation)
        if not useThisSingleColorInPlot is None:
            colorOfGenAtTimePoint = []
        elif not genotypeColorConversion is None:
            geneColorMapperDict = self.createTimePointDependentGeneMapper(resultsTable, genotypeColorConversion)
            colorOfGenAtTimePoint = []
        else:
            colorOfGenAtTimePoint = None
        for tissueName, allGenNames in tissueGeneNameOrdering.items():
            isTissueName = resultsTable[self.tissueColName] == tissueName
            startTissueLocation = startLocation
            if startLocation != 0:
                tissueTicks.append(startTissueLocation - 0.5)
            for geneName in allGenNames:
                isGeneName = resultsTable[self.pureGenotypeColName] == geneName
                startGeneLocation = startLocation
                if startLocation != 0:
                    genTicks.append(startGeneLocation - 0.5)
                isGeneInTissue = isTissueName & isGeneName
                groupedTimePointTables = resultsTable[isGeneInTissue].groupby(self.numericTimePointColName, sort=True)
                for timePointIdx, timePointTable in groupedTimePointTables:
                    timePointName = timePointTable[self.timePointNameColName].unique()
                    if len(timePointName) != 1:
                        warnErrorText = f"There are more than one unique time point names {timePointName} for the time point idx of {timePointIdx} of tissue {tissueName} and genes {geneName} {len(timePointName)} != 1"
                        if enforceTimePointIdHavingOneName:
                            assert len(timePointName) == 1, warnErrorText
                        elif warnAboutTimeIdHavingMoreThanOneName:
                            warnings.warn(warnErrorText)
                    if sortDifferentColumnTypesNextToEachOther and not self.combinedColumnTypeName is None:
                        groupedColumnTypeTables = timePointTable.groupby(self.combinedColumnTypeName, sort=False)
                        for columnName, tableOfColumnType in groupedColumnTypeTables:
                            indicesOfGroup = tableOfColumnType.index
                            orderedTimePoints[indicesOfGroup] = startLocation
                            timePointNames.append(timePointName[0])
                            xOffsets.append(startLocation)
                            entryIdentifier.append((tissueName, geneName, timePointIdx))
                            startLocation += 1
                    else:
                        indicesOfGroup = timePointTable.index
                        orderedTimePoints[indicesOfGroup] = startLocation
                        timePointNames.append(timePointName[0])
                        xOffsets.append(startLocation)
                        entryIdentifier.append((tissueName, geneName, timePointIdx))
                        startLocation += 1
                    if not useThisSingleColorInPlot is None:
                        colorOfGenAtTimePoint.append(useThisSingleColorInPlot)
                    elif not genotypeColorConversion is None:
                        selectedColor = geneColorMapperDict[geneName].to_rgba(timePointIdx)
                        colorOfGenAtTimePoint.append(selectedColor)
                if np.any(isGeneInTissue):
                    allGeneNames.append(geneName)
                    endGeneLocation = startLocation
                    currentGeneLocation = startGeneLocation - 0.5 + 0.5 * (endGeneLocation - startGeneLocation)
                    geneLocations.append(currentGeneLocation)
            if np.any(isTissueName):
                allTissueNames.append(tissueName)
                endTissueLocation = startLocation
                currentTissueLocation = startTissueLocation - 0.5 + 0.5 * (endTissueLocation - startTissueLocation)
                tissueLocations.append(currentTissueLocation)
        tissueTicks.append(startLocation - 1)
        genTicks.append(startLocation - 1)
        if increaseBounds:
            if increaseStartBounds:
                tissueTicks[0] -= 0.5
                genTicks[0] -= 0.5
            if increaseEndBounds:
                tissueTicks[-1] += 0.5
                genTicks[-1] += 0.5
        axisParameter = {"timePointNames": timePointNames, "genTicks": genTicks, "geneLocations": geneLocations, "allGeneNames": allGeneNames,
                         "tissueTicks": tissueTicks, "tissueLocations": tissueLocations, "allTissueNames": allTissueNames, "xOffsets": xOffsets,
                         "colorOfGenAtTimePoint": colorOfGenAtTimePoint, "entryIdentifier": entryIdentifier, "orderedTimePoints": list(orderedTimePoints)}
        resultsTable["orderedTimePoints"] = orderedTimePoints
        return axisParameter, resultsTable

    def removeTableEntriesNotPresentIn(self, resultsTable, tissueGeneNameOrdering):
        isInTissueGeneNameOrderingPresent = np.full(len(resultsTable), False)
        for tissueName, geneNames in tissueGeneNameOrdering.items():
            isTissue = resultsTable[self.tissueColName].to_numpy() == tissueName
            isGene = np.isin(resultsTable[self.pureGenotypeColName], geneNames)
            isCellCurrentTissueGeneCombi = isTissue & isGene
            isInTissueGeneNameOrderingPresent = isInTissueGeneNameOrderingPresent | isCellCurrentTissueGeneCombi
        notPresentRowIndices = np.where(np.invert(isInTissueGeneNameOrderingPresent))[0]
        resultsTable.drop(notPresentRowIndices, inplace=True)
        return resultsTable.reset_index(drop=True)

    def createTimePointDependentGeneMapper(self, resultsTable, genotypeColorConversion, visualiseColors=False):
        allGenNames = resultsTable[self.pureGenotypeColName]
        allUniqueGenNames = allGenNames.unique()
        geneColorMapperDict = {}
        for geneName in allUniqueGenNames:
            timePointsOfGene = resultsTable.loc[geneName == allGenNames, self.numericTimePointColName]
            minimum, maximum = timePointsOfGene.min(), timePointsOfGene.max()
            norm = colors.Normalize(vmin=minimum, vmax=maximum, clip=False)
            baseColor = genotypeColorConversion[geneName]
            lightBaseColor = self.adjust_lightness(baseColor)
            if visualiseColors:
                print("#{:X}{:X}{:X}".format(*[int(i * 255) for i in baseColor]), "#{:X}{:X}{:X}".format(*[int(i * 255) for i in lightBaseColor]))
                import matplotlib.patches as patches
                rect = patches.Rectangle((0, 0), 40, 30, linewidth=1, edgecolor=baseColor, facecolor=baseColor)
                rectLight = patches.Rectangle((0, 0.5), 40, 30, linewidth=1, edgecolor=lightBaseColor, facecolor=lightBaseColor)
                fig, ax = plt.subplots(1, 1)
                ax.add_patch(rect)
                ax.add_patch(rectLight)
                plt.show()
            colorMap = colors.LinearSegmentedColormap.from_list("", [baseColor, lightBaseColor])
            mapper = cm.ScalarMappable(norm=norm, cmap=colorMap)
            geneColorMapperDict[geneName] = mapper
        return geneColorMapperDict

    def adjust_lightness(self, color, amount=0.5):
        try:
            c = colors.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*colors.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

    def poolDictsByExtendingLists(self, dictList, keysToAppend=[]):
        pooledDict = dictList[0]
        for k in keysToAppend:
            pooledDict[k] = [pooledDict[k]]
        for i, currentDict in enumerate(dictList):
            if i == 0:
                continue
            for k, v in currentDict.items():
                if k in pooledDict:
                    if k in keysToAppend:
                        pooledDict[k].append(v)
                    else:
                        pooledDict[k].extend(v)
                else:
                    print("Something went potentially wrong while merging axisParameter dictionaries!!!")
        return pooledDict

    def determineAxisParametersOfCombinedColumnsNextToEachOther(self, resultsTable, tissueGeneNameOrdering, genotypeColorConversion=None,
                                                                increaseBounds=False, increaseStartBounds=True, increaseEndBounds=True,
                                                                warnAboutTimeIdHavingMoreThanOneName=True, enforceTimePointIdHavingOneName=False,
                                                                includeFirstStartLocation=True, startLocation=0,
                                                                sortDifferentColumnTypesNextToEachOther=False, **kwargs):
        numberOfCells = np.sum(len(table) for table in resultsTable)
        orderedTimePoints = np.zeros(numberOfCells, dtype=int)
        resultsTable = pd.concat(resultsTable).reset_index(drop=True)
        timePointNames, entryIdentifier = [], []
        genTicks, geneLocations, allGeneNames = [], [], []
        tissueTicks, tissueLocations, allTissueNames = [], [], []
        if includeFirstStartLocation:
            genTicks.append(startLocation)
            tissueTicks.append(startLocation)
        if not genotypeColorConversion is None:
            geneColorMapperDict = self.createTimePointDependentGeneMapper(resultsTable, genotypeColorConversion)
            colorOfGenAtTimePoint = []
        else:
            colorOfGenAtTimePoint = None
        for tissueName, allGenNames in tissueGeneNameOrdering.items():
            isTissueName = resultsTable[self.tissueColName] == tissueName
            startTissueLocation = startLocation
            if startLocation != 0:
                tissueTicks.append(startTissueLocation - 0.5)
            for geneName in allGenNames:
                isGeneName = resultsTable[self.pureGenotypeColName] == geneName
                startGeneLocation = startLocation
                if startLocation != 0:
                    genTicks.append(startGeneLocation - 0.5)
                isGeneInTissue = isTissueName & isGeneName
                groupedTimePointTables = resultsTable[isGeneInTissue].groupby(self.numericTimePointColName, sort=True)
                for timePointIdx, timePointTable in groupedTimePointTables:

                    timePointName = timePointTable[self.timePointNameColName].unique()
                    if len(timePointName) != 1:
                        warnErrorText = f"There are more than one unique time point names {timePointName} for the time point idx of {timePointIdx} of tissue {tissueName} and genes {geneName} {len(timePointName)} != 1"
                        if enforceTimePointIdHavingOneName:
                            assert len(timePointName) == 1, warnErrorText
                        elif warnAboutTimeIdHavingMoreThanOneName:
                            warnings.warn(warnErrorText)
                    if sortDifferentColumnTypesNextToEachOther and not self.combinedColumnTypeName is None:
                        groupedColumnTypeTables = timePointTable.groupby(self.combinedColumnTypeName, sort=False)
                        for columnName, tableOfColumnType in groupedColumnTypeTables:
                            indicesOfGroup = tableOfColumnType.index
                            orderedTimePoints[indicesOfGroup] = startLocation
                            timePointNames.append(timePointName[0])
                            startLocation += 1
                    else:
                        indicesOfGroup = timePointTable.index
                        orderedTimePoints[indicesOfGroup] = startLocation
                        timePointNames.append(timePointName[0])
                        startLocation += 1
                    if not genotypeColorConversion is None:
                        selectedColor = geneColorMapperDict[geneName].to_rgba(timePointIdx)
                        colorOfGenAtTimePoint.append(selectedColor)
                    entryIdentifier.append((tissueName, geneName, timePointIdx))
                if np.any(isGeneInTissue):
                    allGeneNames.append(geneName)
                    endGeneLocation = startLocation
                    currentGeneLocation = startGeneLocation - 0.5 + 0.5 * (endGeneLocation - startGeneLocation)
                    geneLocations.append(currentGeneLocation)
            if np.any(isTissueName):
                allTissueNames.append(tissueName)
                endTissueLocation = startLocation
                currentTissueLocation = startTissueLocation - 0.5 + 0.5 * (endTissueLocation - startTissueLocation)
                tissueLocations.append(currentTissueLocation)
        tissueTicks.append(startLocation - 1)
        genTicks.append(startLocation - 1)
        if increaseBounds:
            if increaseStartBounds:
                tissueTicks[0] -= 0.5
                genTicks[0] -= 0.5
            if increaseEndBounds:
                tissueTicks[-1] += 0.5
                genTicks[-1] += 0.5
        axisParameter = {"timePointNames": timePointNames, "genTicks": genTicks, "geneLocations": geneLocations, "allGeneNames": allGeneNames,
                         "tissueTicks": tissueTicks, "tissueLocations": tissueLocations, "allTissueNames": allTissueNames,
                         "colorOfGenAtTimePoint": colorOfGenAtTimePoint, "entryIdentifier": entryIdentifier}
        resultsTable["orderedTimePoints"] = orderedTimePoints
        return axisParameter

    def drawVerticalLinesAt(self, givenXTicks, drawThickMidLine=False):
        for x in givenXTicks:
            plt.axvline(x=x, ymin=0.05, ymax=0.95, linestyle="--", alpha=0.6, color="black")
        if drawThickMidLine:
            x = givenXTicks[len(givenXTicks) // 2]
            plt.axvline(x=x, ymin=0.05, ymax=0.95, linestyle="-", color="black")

    def plotAxis(self, ax1, axisParameter, nextAxisYDistance=-0.05, xPos=-0.05, showMinimalXLabels=False):
        timePointRange = np.arange(len(axisParameter["timePointNames"]))
        ax1.set_xlim(np.min(timePointRange) - 0.5, np.max(timePointRange) + 0.5)
        ax1.set_xticks(timePointRange)
        ax1.set_xticklabels(axisParameter["timePointNames"], fontsize="small")
        if showMinimalXLabels:
            return

        ax2 = ax1.twiny()
        ax2.spines['bottom'].set_position(('axes', nextAxisYDistance))
        ax2.tick_params(axis='x', direction='in', which='major')
        ax2.xaxis.set_ticks_position('bottom')
        ax2.xaxis.set_label_position('bottom')

        ax2.set_xlim(ax1.get_xlim())  # same limits
        genTicks = axisParameter["genTicks"]
        ax2.set_xticks(genTicks)
        ax2.spines['bottom'].set_bounds([genTicks[0], genTicks[-1]])
        ax2.xaxis.set_major_formatter(ticker.NullFormatter())
        ax2.xaxis.set_minor_locator(ticker.FixedLocator(axisParameter["geneLocations"]))
        ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(axisParameter["allGeneNames"]))
        ax2.set_xlabel("genotype")

        ax3 = ax2.twiny()
        nextAxisYDistance += nextAxisYDistance

        ax3.spines['bottom'].set_position(('axes', nextAxisYDistance))
        ax3.tick_params(axis='x', direction='in', which='major')
        ax3.xaxis.set_ticks_position('bottom')
        ax3.xaxis.set_label_position('bottom')

        ax3.set_xlim(ax1.get_xlim())  # same datalimits
        tissueTicks = axisParameter["tissueTicks"]
        ax3.set_xticks(tissueTicks)
        ax3.spines['bottom'].set_bounds([tissueTicks[0], tissueTicks[-1]])
        ax3.xaxis.set_major_formatter(ticker.NullFormatter())
        ax3.xaxis.set_minor_locator(ticker.FixedLocator(axisParameter["tissueLocations"]))
        ax3.xaxis.set_minor_formatter(ticker.FixedFormatter(axisParameter["allTissueNames"]))
        ax3.set_xlabel("tissue name")

        ax1.xaxis.set_label_coords(xPos, -0.015)
        ax2.xaxis.set_label_coords(xPos, -0.085)
        ax3.xaxis.set_label_coords(xPos, -0.155)
        tickWidth = 1.5
        ax1.tick_params(which="both", width=tickWidth)
        ax2.tick_params(which="both", width=tickWidth)
        ax3.tick_params(which="both", width=tickWidth)

    def determineNumberOfSamplesForValuesOfColumns(self, table, groupColumnsBy, butShowOnlyFor=None, capitalize=False):
        textOfGroupingToAdd = {}
        if not butShowOnlyFor is None:
            whichGroupColumnToSelect = np.where(np.isin(groupColumnsBy, butShowOnlyFor))[0][0]
        for groupId, groupedTable in table.groupby(groupColumnsBy):
            numberOfSamples = len(groupedTable)
            if not self.combinedColumnTypeName is None:
                nrOfReplicates = len(groupedTable[self.combinedColumnTypeName].unique())
                numberOfSamples = numberOfSamples // nrOfReplicates
            sampleNumberTxt = f"n: {numberOfSamples}"
            if capitalize:
                sampleNumberTxt = sampleNumberTxt.upper()
            if not butShowOnlyFor is None:
                groupId = groupId[whichGroupColumnToSelect]
            textOfGroupingToAdd[groupId] = sampleNumberTxt
        if not butShowOnlyFor is None:
            groupColumnsBy = groupColumnsBy[whichGroupColumnToSelect]
        if type(groupColumnsBy) == list:
            groupColumnsBy = tuple(groupColumnsBy)
        return {groupColumnsBy: textOfGroupingToAdd}

    def drawLinearRegression(self, resultsTable, yColName, ax, axisParameter, xColName="orderedTimePoints",
                             printPooledRegressionResults=True, textOfGroupValuePairsToAdd={}, sepWith="\n"):
        pooledRegressionResults = {"genotype": []}
        if self.pureGenotypeColName in textOfGroupValuePairsToAdd:
            textOfGroupingToAdd = textOfGroupValuePairsToAdd[self.pureGenotypeColName]
        else:
            textOfGroupingToAdd = {}
        for genotype in axisParameter["allGeneNames"]:
            isGenotype = resultsTable[self.pureGenotypeColName] == genotype
            x = resultsTable.loc[isGenotype][xColName]
            y = resultsTable.loc[isGenotype, yColName]
            regressionResults, regressionText = CorrelationDataPlotter().addRegressionLineAndText(ax, x, y, manuallyAddText=True,
                                                                                          showRSquared=True, addLinearFormulaText=False, lw=3)
            # additionally determine text to plot 
            if genotype in textOfGroupingToAdd:
                textOfGroupingToAdd[genotype] += sepWith + regressionText
            else:
                textOfGroupingToAdd[genotype] = regressionText
            # and regression results of genotypes
            pooledRegressionResults["genotype"].append(genotype)
            for key, value in regressionResults.items():
                if key in pooledRegressionResults:
                    pooledRegressionResults[key].append(value)
                else:
                    pooledRegressionResults[key] = [value]
        pooledRegressionResults = pd.DataFrame(pooledRegressionResults)
        if printPooledRegressionResults:
            print(pooledRegressionResults.to_string())
        textOfGroupValuePairsToAdd[self.pureGenotypeColName] = textOfGroupingToAdd
        return textOfGroupValuePairsToAdd, pooledRegressionResults

    def addTextOfGroup(self, textOfGroupValuePairsToAdd, resultsTable, xColName, yColName, ax=None, addAtBottomOfGroup=True, textColor="black"):
        for groupColumns, textOfValuesToAdd in textOfGroupValuePairsToAdd.items():
            allGroupingRows = resultsTable[groupColumns]
            for valueOfGroup, textToAdd in textOfValuesToAdd.items():
                isValueOfGroup = allGroupingRows == valueOfGroup
                x = resultsTable.loc[isValueOfGroup, xColName]
                if addAtBottomOfGroup:
                    assert not ax is None, f"When {addAtBottomOfGroup=} the given axis ax needs to be not None"
                    yPos = self.determineVerticalPosition(ax, aboveXAxis=True, table=resultsTable, measureCol=yColName)
                    textLocation = [np.mean(np.unique(x)), yPos]
                else:
                    textLocation = self.determineMiddleTextLocation(x, resultsTable[yColName])
                plt.text(*textLocation, textToAdd,
                         horizontalalignment='center', size='small', color=textColor)

    def determineVerticalPosition(self, ax, aboveXAxis=False, table=None, measureCol=None, yOffsetFactor=0.03):
        yLim = ax.get_ylim()
        yOffset = yOffsetFactor * (np.abs(np.max(yLim)) - np.abs(np.min(yLim)))
        if aboveXAxis:
            yPos = np.min(yLim) + yOffset
        else:
            minValue = table[measureCol].min()
            yPos = minValue + yOffset
        return yPos

    def determineMiddleTextLocation(self, xValues, yValues):
        xLim = [np.min(xValues), np.max(xValues)]
        yLim = [np.min(yValues), np.max(yValues)]
        yMean = np.mean(yValues)
        xTextLocation = xLim[0] + 0.5 * (xLim[1] - xLim[0])
        yTextLocationOffset = 0.2 * (yLim[1] - yLim[0])
        yTextLocation = yMean + yTextLocationOffset
        textLocation = (xTextLocation, yTextLocation)
        return textLocation

def mainCreateBoxPlotOf(measureName="lengthGiniCoeff", plotSelectedColumnsAdjacent=None, tissueScenarioNames=["Eng2021Cotyledons", "full cotyledons", "Matz2022SAM"],
                        baseResultsFolder="Results/", resultsFolderExtensions="regularityResults/", measuresBaseName="combinedMeasures_{}.csv",
                        testVsValue=None, saveFigure=True, tissueGeneNameOrdering={"SAM": ["WT"], "cotyledon": ["WT", "WT+Oryzalin", "$\it{ktn1}$-$\it{2}$"]},
                        fixxedYLocator=None, addHLineAt=None, resultsNameExtension="", onlyShowLastAndFirstTimePoint=False, furtherRemoveScenarioIds=None,
                        onlyPlotXLabel=None, reduceResultsToOnePerTissue=False, filenameToSave=None, overwriteTimePointRenamingWith=None, **kwargs):
    colorPalette = sns.color_palette("colorblind")
    genotypeColorConversion = {"WT": colorPalette[7], "WT+Oryzalin": colorPalette[8], "ktn": colorPalette[0], "$\it{ktn1}$-$\it{2}$": colorPalette[0]}
    resultsTableFilenames = [f"{baseResultsFolder}{measuresBaseName.format(name)}" for name in tissueScenarioNames]
    renameTimePointDict = {"T0": "0-96h", "T1": "0-96h", "T2": "0-96h", "T3": "0-96h", "T4": "0-96h"} #, "T0": "0h", "T1": "24h", "T2": "48h", "T3": "72h", "T4":"96h"} #
    if not overwriteTimePointRenamingWith is None:
        for timePointName, renameTo in overwriteTimePointRenamingWith.items():
            renameTimePointDict[timePointName] = renameTo
    timePointConversionFromStrToNumber = {"0h": 0, '12h': 12, "24h": 24, '36h': 36, "48h": 48, '60h': 60, "72h": 72, '84h': 84, "96h": 96, '108h': 108, "120h": 120, '132h': 132, '144h': 144, "0-96h":0, "T0": 0, "T1": 24, "T2": 48, "T3": 72, "T4":96}
    renameGenotypesDict = {"WT inflorescence meristem": "WT SAM", "col-0": "WT cotyledon", "WT": "WT cotyledon",
                           "Oryzalin": "WT+Oryzalin cotyledon", "ktn1-2": "$\it{ktn1}$-$\it{2}$ cotyledon"}
    if saveFigure:
        if filenameToSave is None:
            filenameToSave = baseResultsFolder + resultsFolderExtensions + measureName + resultsNameExtension + ".png"
    else:
        filenameToSave = filenameToSave
    myBasePlotter = ExtendedPlotter(resultsTableFilenames, timePointConversionFromStrToNumber, renameGenotypesDict=renameGenotypesDict,
                                    renameTimePointDict=renameTimePointDict, onlyShowLastAndFirstTimePoint=onlyShowLastAndFirstTimePoint,
                                    furtherRemoveScenarioIds=furtherRemoveScenarioIds, reduceResultsToOnePerTissue=reduceResultsToOnePerTissue)
    if plotSelectedColumnsAdjacent is None:
        myBasePlotter.CreateBoxPlot(measureName, tissueGeneNameOrdering, filenameToSave=filenameToSave,
                                    genotypeColorConversion=genotypeColorConversion, onlyPlotXLabel=onlyPlotXLabel,
                                    testVsValue=testVsValue, fixxedYLocator=fixxedYLocator, addHLineAt=addHLineAt, **kwargs)
    else:
        myBasePlotter.StackColumnsOfResultsTableVertically(plotSelectedColumnsAdjacent, newColumnName=measureName)
        myBasePlotter.CreateBoxPlot(measureName, tissueGeneNameOrdering, filenameToSave=filenameToSave,
                                    genotypeColorConversion=genotypeColorConversion, onlyPlotXLabel=onlyPlotXLabel,
                                    testVsValue=testVsValue, fixxedYLocator=fixxedYLocator, addHLineAt=addHLineAt, **kwargs)

def plotPatchyCotyledonResults(plotAreaMeasures=False, plotRegularityMeasures=False,
                              doIgnoreGuardCells=False, plotFullDataSet=False, plotCombinedDataSet=False,
                              allDataFontSize=40, reducedFontSize=45):
    patchyCotyledonOrdering = {"cotyledon": ["WT", "WT+Oryzalin", "$\it{ktn1}$-$\it{2}$"]}
    tissueComparisonOrdering = {"cotyledon": ["WT"], "SAM": ["WT"]}
    if plotAreaMeasures:
        print("Plotting area measures")
        if plotCombinedDataSet:
            mainCreateBoxPlotOf(measureName="Ratio", plotSelectedColumnsAdjacent=["ratio_originalPolygonArea_labelledImageArea", "ratio_regularPolygonArea_labelledImageArea"],
                                resultsNameExtension="_WtTissueComparison", tissueScenarioNames=["full cotyledons", "Matz2022SAM"], tissueGeneNameOrdering=tissueComparisonOrdering,
                                compareAllAgainstAll=True, drawLinesAtTicksParameter="genTicks", showMinimalXLabels=True, excludeYLabel=True,
                                resultsFolderExtensions="ratioResults/", testVsValue=1, addHLineAt=1, fontSize=reducedFontSize)
        if plotFullDataSet:
            mainCreateBoxPlotOf(measureName="ratio_originalPolygonArea_labelledImageArea", resultsFolderExtensions="ratioResults/",
                                testVsValue=1, addHLineAt=1, fontSize=allDataFontSize, xPos=-0.035,
                                tissueScenarioNames=["Eng2021Cotyledons"], tissueGeneNameOrdering=patchyCotyledonOrdering,
                                showMinimalXLabels=True, addLinearRegression=True, ylim=(0.4542479550844027, 1.946669865152332))
            mainCreateBoxPlotOf(measureName="ratio_regularPolygonArea_labelledImageArea", resultsFolderExtensions="ratioResults/",
                                testVsValue=1, addHLineAt=1, fontSize=allDataFontSize, xPos=-0.035,
                                tissueScenarioNames=["Eng2021Cotyledons"], tissueGeneNameOrdering=patchyCotyledonOrdering,
                                showMinimalXLabels=True, addLinearRegression=True, ylim=(0.4542479550844027, 1.946669865152332))

    if plotRegularityMeasures:
        print("Plotting regularity measures")
        if plotCombinedDataSet:
            mainCreateBoxPlotOf(measureName="Gini coefficient of ", plotSelectedColumnsAdjacent=["lengthGiniCoeff", "angleGiniCoeff"], resultsFolderExtensions="regularityResults/",
                                resultsNameExtension="_WtTissueComparison", tissueScenarioNames=["full cotyledons", "Matz2022SAM"], tissueGeneNameOrdering=tissueComparisonOrdering,
                                compareAllAgainstAll=True, drawLinesAtTicksParameter="genTicks", showMinimalXLabels=True, excludeYLabel=True, fontSize=reducedFontSize)
        if plotFullDataSet:
            mainCreateBoxPlotOf(measureName="lengthGiniCoeff",  resultsFolderExtensions="regularityResults/",
                                fontSize=allDataFontSize, xPos=-0.035,  showMinimalXLabels=True, excludeYLabel=True,
                                tissueScenarioNames=["Eng2021Cotyledons"], tissueGeneNameOrdering=patchyCotyledonOrdering, addLinearRegression=True)
            mainCreateBoxPlotOf(measureName="angleGiniCoeff",  resultsFolderExtensions="regularityResults/",
                                fontSize=allDataFontSize, xPos=-0.035,  showMinimalXLabels=True, excludeYLabel=True,
                                tissueScenarioNames=["Eng2021Cotyledons"], tissueGeneNameOrdering=patchyCotyledonOrdering, addLinearRegression=True)
        if doIgnoreGuardCells:
            mainCreateBoxPlotOf(measureName="angleGiniCoeff_ignoringGuardCells", resultsNameExtension="_reducedTimePoints", resultsFolderExtensions="regularityResults/",
                                compareAllAgainstAll=True, fontSize=reducedFontSize, yLabePad=4, showMinimalXLabels=True,
                                tissueScenarioNames=["Eng2021Cotyledons"], tissueGeneNameOrdering=patchyCotyledonOrdering, addLinearRegression=True)
            mainCreateBoxPlotOf(measureName="lengthGiniCoeff_ignoringGuardCells", resultsNameExtension="_reducedTimePoints", resultsFolderExtensions="regularityResults/",
                                compareAllAgainstAll=True, fontSize=reducedFontSize, yLabePad=4, showMinimalXLabels=True,
                                tissueScenarioNames=["Eng2021Cotyledons"], tissueGeneNameOrdering=patchyCotyledonOrdering, addLinearRegression=True)

if __name__ == '__main__':
    plotPatchyCotyledonResults(plotAreaMeasures=True, plotRegularityMeasures=True,
                               plotCombinedDataSet=True, plotFullDataSet=True, doIgnoreGuardCells=False)
