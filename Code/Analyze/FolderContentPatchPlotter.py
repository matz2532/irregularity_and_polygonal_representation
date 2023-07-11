import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, "./Code/DataStructures/")
from CellIdTracker import CellIdTracker
from FolderContent import FolderContent
from MultiFolderContent import MultiFolderContent
from PatchCreator import PatchCreator
from pathlib import Path

class FolderContentPatchPlotter (PatchCreator):

    def __init__(self):
        pass

    def combinedPatchesToFigure(self, allEntryIdentifiersPlusFolderContents, measureDataFilenameKey, selectedSubMeasure, saveAsFilename=None,
                                surfaceContourPerCellFilenameKey="orderedJunctionsPerCellFilename", overlaidContourEdgePerCellFilenameKey=None, removeNotSelectedContourCells=True,
                                colorBarLabel=None, colorMap=None, setAxesParameterInSinglePlots=True, overwritingMeasureDataKwargs=None,
                                usePooledValueRange=False, selectRowIdxFromDict=None, useAbsMaxRange=False, defaultshowColorBar=None,
                                figAxesParameterDict=False, nrOfRows=1, nrOfCols=1, ax=None, colSizeInInches=None, showTitle=True, fontSize=45):
        if not measureDataFilenameKey is None:
            measureDataFilenameKey = self.measureDataFilenameKey
        if usePooledValueRange:
            allMinMax = []
            for entryIdentifier in allEntryIdentifiersPlusFolderContents:
                measureData = self.extractMeasureDataFrom(entryIdentifier, selectedSubMeasure, overwritingMeasureDataKwargs)
                allMinMax.append([np.min(measureData), np.max(measureData)])
            allMinMax = np.asarray(allMinMax)
            if useAbsMaxRange:
                absMax = np.max(np.abs(allMinMax))
                absMax=2
                colorMapValueRange = [- absMax, absMax]
            else:
                colorMapValueRange = [np.min(allMinMax[:, 0]), np.max(allMinMax[:, 1])]
        else:
            colorMapValueRange = None
        plt.rcParams["font.size"] = fontSize
        plt.rcParams["font.family"] = "Arial"
        fig = None
        if figAxesParameterDict:
            tissueIdentifierNames = ["_".join(tissueIdentifierEntries[:-1]) for tissueIdentifierEntries in allEntryIdentifiersPlusFolderContents]
            isTissueAxesParameterPresent = [tissueName in allAxesParameter for tissueName in tissueIdentifierNames]
            if np.all(isTissueAxesParameterPresent):
                if colSizeInInches is None:
                    defaultFigSize = plt.rcParams["figure.figsize"]
                    figsize = [nrOfCols * defaultFigSize[0], nrOfRows * defaultFigSize[0]]
                    if showTitle:
                        figsize[1] += 0.5 * defaultFigSize[0]
                    if not colorMapValueRange is None:
                        figsize[0] += 0.2 * defaultFigSize[0]
                else:
                    rowSizeInInches = nrOfRows * colSizeInInches / nrOfCols
                    figsize = [colSizeInInches, rowSizeInInches]
                fig = plt.figure(figsize = figsize)
                figAxesParameterDict = {"fig": fig, "nrOfRows": nrOfRows, "nrOfCols": nrOfCols, "currentAxes": 0}
            else:
                missingTissueNames = np.array(tissueIdentifierNames)[np.invert(isTissueAxesParameterPresent)]
                missingTissuesText = '\n'.join(missingTissueNames)
                print(f"The tissue image section parameter is missing for\n{missingTissuesText}\nThe following entries are present {list(allAxesParameter.keys())}\nPress space after aligning the axes view, copy the resulting text into the 'allAxesParameter' dictionary, and rerun the code.")
                figAxesParameterDict = False
        for i, entryIdentifier in enumerate(allEntryIdentifiersPlusFolderContents):#
            if figAxesParameterDict:
                figAxesParameterDict["currentAxes"] = i + 1
            if entryIdentifier[-1] == "Images/Eng2021Cotyledons.pkl" and measureDataFilenameKey == "polygonalGeometricData":
                resultsTable = pd.read_csv("Results/combinedMeasures_Eng2021Cotyledons.csv")
            else:
                resultsTable = None
            if defaultshowColorBar is None:
                if not colorMapValueRange is None:
                    showColorBar = i == len(allEntryIdentifiersPlusFolderContents) - 1
                else:
                    showColorBar = False
            else:
                showColorBar = defaultshowColorBar
            self.plotPatchesOf(entryIdentifier, surfaceContourPerCellFilenameKey, measureDataFilenameKey, overwritingMeasureDataKwargs=overwritingMeasureDataKwargs,
                               overlaidContourEdgePerCellFilenameKey=overlaidContourEdgePerCellFilenameKey, removeNotSelectedContourCells=removeNotSelectedContourCells,
                               allAxesParameter=allAxesParameter, selectedSubMeasure=selectedSubMeasure, tableWithValues=resultsTable,
                               figAxesParameterDict=figAxesParameterDict, setAxesParameterInSinglePlots=setAxesParameterInSinglePlots, showTitle=showTitle,
                               genotypeToScenarioName=genotypeToScenarioName, timePointToName=timePointToName,
                               colorBarLabel=colorBarLabel, colorMapValueRange=colorMapValueRange, colorMap=colorMap, showColorBar=showColorBar, fig=fig, ax=ax)
        if not saveAsFilename is None:
            plt.savefig(saveAsFilename, bbox_inches="tight", dpi=300)
            plt.close()
        elif figAxesParameterDict:
            plt.show()

    def extractMeasureDataFrom(self, entryIdentifier, selectedSubMeasure: str, overwritingMeasureDataKwargs: dict = None):
        allFolderContentsFilename = entryIdentifier[-1]
        multiFolderContent = MultiFolderContent(allFolderContentsFilename)
        folderContent = multiFolderContent.GetFolderContentOfIdentifier(entryIdentifier[:3])
        measureData = folderContent.LoadKeyUsingFilenameDict(self.measureDataFilenameKey, skipfooter=4)
        if isinstance(measureData, dict):
            if selectedSubMeasure in measureData:
                if overwritingMeasureDataKwargs is None:
                    measureData = list(measureData[selectedSubMeasure].values())
                else:
                    if "keyToSelect" in overwritingMeasureDataKwargs:
                        keyToSelect = overwritingMeasureDataKwargs["keyToSelect"]
                        measureData = measureData[keyToSelect]
                    if "valuesIdx" in overwritingMeasureDataKwargs:
                        measureData = np.array(measureData).T
                        measureData = measureData[:, overwritingMeasureDataKwargs["valuesIdx"]]
            else:
                measureData = list(measureData.values())
        else:
            measureData = measureData.iloc[:, 1]
        return measureData

    def plotPatchesOf(self, entryIdentifier, surfaceContourPerCellFilenameKey, measureDataFilenameKey=None, showPlot=False,
                      overwritingMeasureDataKwargs=None, overlaidContourEdgePerCellFilenameKey=None, overlaidContourOffset=None, offsetOfContours=None,
                      selectedCellLabels=None, convertFromIdToLabel=True, plotOutlines=True, isThreeDimensional=True,
                      allAxesParameter={}, selectedSubMeasure=None, tableWithValues=None, colorBarLabel=None, faceColorDict=None,
                      figAxesParameterDict={}, setAxesParameterInSinglePlots=True, getCellIdByKeyStroke=False,
                      backgroundFilename=None, loadBackgroundFromKey=None, is3DBackground=False, backgroundImageOffset=None,
                      showTitle=False, scaleBarSize=None, genotypesResolutionDict=None, scaleBarOffset=None, fig=None, ax=None,
                      genotypeToScenarioName={}, timePointToName={}, colorMapValueRange=None, colorMapper=None, colorMap=None,
                      visualiseAbsDiffToMeanOfValues=False, showColorBar=True, removeNotSelectedContourCells=True, patchKwargs={}):
        allFolderContentsFilename = entryIdentifier[-1]
        multiFolderContent = MultiFolderContent(allFolderContentsFilename)
        folderContent = multiFolderContent.GetFolderContentOfIdentifier(entryIdentifier[:3])
        contours = folderContent.LoadKeyUsingFilenameDict(surfaceContourPerCellFilenameKey, convertDictKeysToInt=True, convertDictValuesToNpArray=True)
        if overlaidContourEdgePerCellFilenameKey is None:
            polygonalOutlineDict = None
        else:
            polygonalOutlineDict = folderContent.LoadKeyUsingFilenameDict(overlaidContourEdgePerCellFilenameKey)
            if not overlaidContourOffset is None:
                for cellLabel in polygonalOutlineDict.keys():
                    polygonalOutlineDict[cellLabel] += overlaidContourOffset
        if not offsetOfContours is None:
            for cellLabel in contours.keys():
                contours[cellLabel] += offsetOfContours
        if not tableWithValues is None:
            genotype, replicateId, timePoint = folderContent.GetTissueInfos()
            isGenotype = genotype == tableWithValues.iloc[:, 0]
            isReplicate = replicateId == tableWithValues.iloc[:, 1]
            isTimePoint = timePoint == tableWithValues.iloc[:, 2]
            isTissue = isGenotype & isReplicate & isTimePoint
            measureData = tableWithValues.loc[isTissue, ["cell label", "originalPolygonArea"]].copy()
            measureData.reset_index(drop=True)
        else:
            if measureDataFilenameKey is None:
                measureData = None # allows to plot empty cells
            else:
                measureData = folderContent.LoadKeyUsingFilenameDict(measureDataFilenameKey, skipfooter=4)
        if isinstance(measureData, dict):
            if overwritingMeasureDataKwargs is None:
                if isinstance(measureData[list(measureData.keys())[0]], dict):
                    if "ratio" in selectedSubMeasure:
                        measureData = self.calculateRatioMeasureData(measureData, selectedSubMeasure, entryIdentifier)
                    else:
                        measureData = self.extractSubMeasureOfCellsFromDict(selectedSubMeasure, measureData, entryIdentifier)
            else:
                measureData, measureDataArray, faceColorDict = self.determineMeasureData(measureData, overwritingMeasureDataKwargs, faceColorDict=faceColorDict)
            if visualiseAbsDiffToMeanOfValues:
                mean = np.mean(list(measureData.values()))
                for k, v in measureData.items():
                    measureData[k] = np.abs(v-mean)
        significantCells = None
        visualizePValue = False
        if not overwritingMeasureDataKwargs is None:
            if "labelsIdx" in overwritingMeasureDataKwargs and "pValueIdx" in overwritingMeasureDataKwargs:
                if "visualizePValue" in overwritingMeasureDataKwargs:
                    visualizePValue = overwritingMeasureDataKwargs["visualizePValue"]
                else:
                    visualizePValue = True
        if visualizePValue:
            labels = measureDataArray[:, overwritingMeasureDataKwargs["labelsIdx"]]
            pValueIdx = overwritingMeasureDataKwargs["pValueIdx"]
            alpha = 0.05
            pValues = measureDataArray[:, pValueIdx]
            isSignificant = pValues < alpha
            if np.any(isSignificant):
                significantCells = labels[isSignificant]
        tissueName = folderContent.GetTissueName()
        if tissueName in allAxesParameter:
            axesParameter = allAxesParameter[tissueName]
            # if not figAxesParameterDict and setAxesParameterInSinglePlots:
            #     return
        else:
            axesParameter = None
        if len(genotypeToScenarioName) > 0 and len(timePointToName) > 0 and showTitle:
            if entryIdentifier[0] in genotypeToScenarioName:
                scenarioName = genotypeToScenarioName[entryIdentifier[0]]
            else:
                scenarioName = entryIdentifier[0]
            timePointName = entryIdentifier[2]
            if timePointName in timePointToName:
                timePointName = timePointToName[timePointName]
            title = f"{scenarioName} at {timePointName}"
        else:
            title = None
        if ax is None:
            if figAxesParameterDict:
                if isThreeDimensional:
                    projection = "3d"
                else:
                    projection = None
                ax = figAxesParameterDict["fig"].add_subplot(figAxesParameterDict["nrOfRows"], figAxesParameterDict["nrOfCols"],
                                                             figAxesParameterDict["currentAxes"], projection=projection)
            else:
                ax = None
        if not selectedCellLabels is None:
            if convertFromIdToLabel:
                if folderContent.IsKeyInFilenameDict("contourFilename"):
                    myCellIdTracker = CellIdTracker()
                    myCellIdTracker.RunCellIdTracker(contourFile=folderContent.GetFilenameDictKeyValue("contourFilename"))
                    labelsToIdConverter = myCellIdTracker.GetIdsToLabelsDict()
                    selectedCellLabels = [labelsToIdConverter[id] for id in selectedCellLabels]
                else:
                    print(f"""Attention: The tissue's {folderContent.GetTissueName()} cell labels were not converted. 
                    In case you have provided the id's which should be converted to labels of the tissue,
                    you needs to provide the contourFilename key in the folder content.""")
            cellLabelsOfContours = list(contours.keys())
            for cellLabel in cellLabelsOfContours:
                if not cellLabel in selectedCellLabels:
                    contours.pop(cellLabel)
                    if type(measureData) == dict:
                        measureData.pop(cellLabel)
                    if removeNotSelectedContourCells:
                        if type(polygonalOutlineDict) == dict:
                            polygonalOutlineDict.pop(cellLabel)
        if not scaleBarSize is None and not genotypesResolutionDict is None:
            genotype = folderContent.GetGenotype()
            if genotype in genotypesResolutionDict:
                scaleBarSize /= genotypesResolutionDict[genotype]
        if not loadBackgroundFromKey is None:
            backgroundFilename = folderContent.GetFilenameDictKeyValue(loadBackgroundFromKey)

        return self.Plot3DPatchesFromOutlineDictOn(contours, ax=ax, measureData=measureData, plotOutlines=plotOutlines, polygonalOutlineDict=polygonalOutlineDict, isThreeDimensional=isThreeDimensional,
                                            axesParameter=axesParameter, showPlot=showPlot, markCells=significantCells, tissueName=tissueName,
                                            title=title, colorBarLabel=colorBarLabel, colorMapValueRange=colorMapValueRange, faceColorDict=faceColorDict,
                                            backgroundFilename=backgroundFilename, is3DBackground=is3DBackground, backgroundImageOffset=backgroundImageOffset,
                                            colorMapper=colorMapper, colorMap=colorMap, showColorBar=showColorBar, scaleBarSize=scaleBarSize, scaleBarOffset=scaleBarOffset, getCellIdByKeyStroke=getCellIdByKeyStroke,
                                            **patchKwargs)

    def calculateRatioMeasureData(self, measureData, selectedSubMeasure, entryIdentifier=None):
        nominatorSubMeasureKey = selectedSubMeasure["ratio"][0]
        denominatorSubMeasureKey = selectedSubMeasure["ratio"][1]
        nominatorData = self.extractSubMeasureOfCellsFromDict(nominatorSubMeasureKey, measureData, entryIdentifier=entryIdentifier)
        denominatorData = self.extractSubMeasureOfCellsFromDict(denominatorSubMeasureKey, measureData, entryIdentifier=entryIdentifier)
        measureData = {}
        for cellId, nominatorValue, in nominatorData.items():
            if cellId in denominatorData:
                denominatorValue = denominatorData[cellId]
                measureData[cellId] = nominatorValue / denominatorValue
            else:
                print(f"Cell id {cellId} is only present as a nominator, but not denominator for {selectedSubMeasure=}")
        return measureData

    def extractSubMeasureOfCellsFromDict(self, selectedSubMeasure, measureData, entryIdentifier=None):
        assert not selectedSubMeasure is None, f"When the values to use for coloring are of type dict you need to further specify which sub measure to select for the tissue {entryIdentifier}, of the value dict key {measureDataFilenameKey}"
        return measureData[selectedSubMeasure]

    def determineMeasureData(self, measureData, overwritingMeasureDataKwargs, faceColorDict=None):
        if "keyToSelect" in overwritingMeasureDataKwargs:
            keyToSelect = overwritingMeasureDataKwargs["keyToSelect"]
            measureData = measureData[keyToSelect]
        if "labelsIdx" in overwritingMeasureDataKwargs and "valuesIdx" in overwritingMeasureDataKwargs:
            measureDataArray = np.array(measureData).T
            labels = measureDataArray[:, overwritingMeasureDataKwargs["labelsIdx"]]
            values = measureDataArray[:, overwritingMeasureDataKwargs["valuesIdx"]]
            if "valueToColorConvert" in overwritingMeasureDataKwargs and ("alpha" in overwritingMeasureDataKwargs or "alphaRange" in overwritingMeasureDataKwargs) and "pValueIdx" in overwritingMeasureDataKwargs:
                pValues = measureDataArray[:, overwritingMeasureDataKwargs["pValueIdx"]]
                if "alphaRange" in overwritingMeasureDataKwargs:
                    upperAlpha, alpha = overwritingMeasureDataKwargs["alphaRange"]
                    norm = colors.Normalize(vmin=upperAlpha, vmax=alpha, clip=True)
                else:
                    alpha = overwritingMeasureDataKwargs["alpha"]
                    norm = None
                isSign = np.array(pValues < alpha)

                values *= isSign
                valueToColorConvert = overwritingMeasureDataKwargs["valueToColorConvert"]
                faceColorDict = {}
                for cellLabel, cellValue, cellPValue in zip(labels, values, pValues):
                    color = valueToColorConvert[cellValue]
                    if not norm is None:
                        if cellPValue < alpha:
                            normalisedValue = int(100 + 155 * (1 - norm(cellPValue)))

                            alphaAsString = f"{normalisedValue:02x}"
                            color += alphaAsString
                    faceColorDict[cellLabel] = color
            measureData = dict(zip(labels, values))
        return measureData, measureDataArray, faceColorDict

allAxesParameter = {"WT inflorescence meristem_P2_T0": {'xlim': [32.66, 58.86], 'ylim': [31.96, 55.71], 'zlim': [8.03, 13.82], 'azim': 100.04, 'elev': 87.77},
                    "WT inflorescence meristem_P2_T4": {'xlim': [22.69, 42.45], 'ylim': [27.63, 46.72], 'zlim': [11.84, 18.83], 'azim': 87.56, 'elev': 82.6},
                    "col-0_20170327 WT S1_0h": {'xlim': [125.79, 379.21], 'ylim': [143.36, 380.64], 'zlim': [1.2, 3.8], 'azim': -90, 'elev': 90},
                    "col-0_20170327 WT S1_96h": {'xlim': [269.27, 772.73], 'ylim': [300.89, 833.11], 'zlim': [1.22, 3.78], 'azim': -90, 'elev': 90},
                    "WT_20200220 WT S1_120h": {'xlim': [216.08, 307.92], 'ylim': [300.86, 424.89], 'zlim': [2.05, 2.95], 'azim': -90, 'elev': 90},
                    "Oryzalin_20170731 oryzalin S2_0h": {'xlim': [131.13, 375.87], 'ylim': [87.56, 275.44], 'zlim': [1.18, 3.82], 'azim': -90, 'elev': 90},
                    "Oryzalin_20170731 oryzalin S2_96h": {'xlim': [223.76, 696.24], 'ylim': [199.91, 586.09], 'zlim': [1.15, 3.85], 'azim': -90, 'elev': 90},
                    "ktn1-2_20180618 ktn1-2 S1_96h": {'xlim': [483.35, 1383.65], 'ylim': [533.94, 1385.06], 'zlim': [1.19, 3.81], 'azim': -90, 'elev': 90},
                    "ktn1-2_20180618 ktn1-2 S2_0h": {'xlim': [213.17, 620.83], 'ylim': [136.24, 411.76], 'zlim': [1.22, 3.78], 'azim': -90, 'elev': 90},
                    "ktn1-2_20180618 ktn1-2 S2_96h": {'xlim': [549.68, 1633.32], 'ylim': [423.7, 1187.3], 'zlim': [1.2, 3.8], 'azim': -90, 'elev': 90},
                    "ktn1-2_20180618 ktn1-2 S6_96h": {'xlim': [487.67, 1473.33], 'ylim': [585.33, 1648.67], 'zlim': [1.2, 3.8], 'azim': -90, 'elev': 90},
                    "ktn1-2_20200220 ktn1-2 S1_120h": {'xlim': [231.75, 350.25], 'ylim': [256.28, 380.47], 'zlim': [1.99, 3.01], 'azim': -90, 'elev': 90},
                    "ktn1-2_20200220 ktn1-2 S2_120h": {'xlim': [106.45, 482.05], 'ylim': [120.27, 511.48], 'zlim': [-0.67, 0.67], 'azim': 120, 'elev': 50},
                    "ktn1-2_20200220 ktn1-2 S3_120h": {'xlim': [130.86, 502.09], 'ylim': [123.32, 470.98], 'zlim': [-0.65, 0.62], 'azim': 120, 'elev': 50},
                    "WT_20200221 WT S2_120h": {'xlim': [95.22, 306.8], 'ylim': [92.95, 306.13], 'zlim': [1.05, 3.95], 'azim': -90, 'elev': 90},
                    "WT_20200221 WT S3_120h": {'xlim': [166.91, 549.05], 'ylim': [128.19, 409.82], 'zlim': [1.05, 3.95], 'azim': -90, 'elev': 90},
                    "WT_20200221 WT S5_120h": {'xlim': [152.36, 471.47], 'ylim': [128.55, 403.5], 'zlim': [1.07, 3.93], 'azim': -90, 'elev': 90},
                    "WT_20200221 WT S2_120h": {'xlim': [95.31, 307.63], 'ylim': [88.85, 302.77], 'zlim': [1.04, 3.96], 'azim': -90, 'elev': 90},
                    "WT_20200221 WT S3_120h": {'xlim': [169.83, 556.0], 'ylim': [120.19, 404.79], 'zlim': [1.03, 3.97], 'azim': -90, 'elev': 90},
                    "WT_20200221 WT S5_120h": {'xlim': [155.2, 479.6], 'ylim': [122.51, 402.02], 'zlim': [1.04, 3.96], 'azim': -90, 'elev': 90},
                    "WT_20200221 WT S2_120h": {'xlim': [97.3, 310.68], 'ylim': [91.1, 306.1], 'zlim': [1.04, 3.96], 'azim': -90, 'elev': 90},
                    "WT_20200221 WT S3_120h": {'xlim': [174.16, 558.99], 'ylim': [118.87, 402.48], 'zlim': [1.04, 3.96], 'azim': -90, 'elev': 90},
                    "WT_20200221 WT S5_120h": {'xlim': [147.7, 477.72], 'ylim': [116.65, 401.01], 'zlim': [1.02, 3.98], 'azim': -90, 'elev': 90},
                    "WT_20200221 WT S2_120h": {'xlim': [91.03, 308.05], 'ylim': [88.4, 307.05], 'zlim': [1.01, 3.99], 'azim': -90, 'elev': 90},
                    "WT_20200221 WT S3_120h": {'xlim': [171.49, 548.12], 'ylim': [126.06, 403.63], 'zlim': [1.07, 3.93], 'azim': -90, 'elev': 90},
                    "WT_20200221 WT S5_120h": {'xlim': [150.84, 471.02], 'ylim': [129.73, 405.6], 'zlim': [1.06, 3.94], 'azim': -90, 'elev': 90},
                    "ktn1-2_20200220 ktn1-2 S2_120h": {'xlim': [112.22, 480.55], 'ylim': [120.31, 503.95], 'zlim': [-0.65, 0.67], 'azim': 120, 'elev': 50},
                    "ktn1-2_20200220 ktn1-2 S3_120h": {'xlim': [130.34, 506.09], 'ylim': [116.14, 468.03], 'zlim': [-0.64, 0.64], 'azim': 120, 'elev': 50},
                    "ktn1-2_20200220 ktn1-2 S2_120h": {'xlim': [105.64, 482.86], 'ylim': [119.42, 512.33], 'zlim': [-0.67, 0.67], 'azim': 120, 'elev': 50},
                    "ktn1-2_20200220 ktn1-2 S3_120h": {'xlim': [128.65, 503.74], 'ylim': [121.5, 472.77], 'zlim': [-0.65, 0.63], 'azim': 120, 'elev': 50},
                    }

genotypeToScenarioName = {"WT inflorescence meristem": "WT SAM", "col-0": "WT cotyledon", "WT": "WT cotyledon", "Oryzalin": "WT+Oryzalin cotyledon", "ktn1-2": "$\it{ktn1}$-$\it{2}$ cotyledon"}
timePointToName = {"T0": "T0", "84h": "84h", "96h": "96h", "120h": "120h", '1DAI': "24h", '1.5DAI': "32h", '2DAI': "48h", '2.5DAI': "60h", '3DAI': "72h", '3.5DAI': "84h", '4DAI': "96h", '4.5DAI': "108h", '5DAI': "120h", '5.5DAI': "132h",
                   '6DAI': "144h"}
colorBarLabelConverter = {"measureData": "polygonal area [µm²]", "polygonalGeometricData": "polygonal area [µm²]", "labelledImageArea":"area [µm²]",
                          "lengthGiniCoeff": "Gini Coefficient of length", "angleGiniCoeff": "Gini Coefficient of angle"}

def mainVisualiseFirstAndLastCotyledons():
    orderedJunctionsPerCellFilenameKey = "orderedJunctionsPerCellFilename"
    measureDataFilenameKey = "regularityMeasuresFilename"
    selectedSubMeasure = "angleGiniCoeff" # "lengthGiniCoeff"  #
    showAllTogether = True
    setAxesParameterInSinglePlots = True

    allEntryIdentifiersPlusFolderContents = [
        ["col-0", "20170327 WT S1", "0h", "Images/Eng2021Cotyledons/Eng2021Cotyledons.pkl"],
        ["Oryzalin", "20170731 oryzalin S2", "0h", "Images/Eng2021Cotyledons/Eng2021Cotyledons.pkl"],
        ["ktn1-2", "20180618 ktn1-2 S2", "0h", "Images/Eng2021Cotyledons/Eng2021Cotyledons.pkl"],
        ["col-0", "20170327 WT S1", "96h", "Images/Eng2021Cotyledons/Eng2021Cotyledons.pkl"],
        ["Oryzalin", "20170731 oryzalin S2", "96h", "Images/Eng2021Cotyledons/Eng2021Cotyledons.pkl"],
        ["ktn1-2", "20180618 ktn1-2 S2", "96h", "Images/Eng2021Cotyledons/Eng2021Cotyledons.pkl"],
    ]
    nrOfRows = 2
    nrOfCols = np.ceil(len(allEntryIdentifiersPlusFolderContents) / nrOfRows)
    colorBarLabel = colorBarLabelConverter[selectedSubMeasure]
    if showAllTogether:
        defaultFigSize = plt.rcParams["figure.figsize"]
        figsize = (nrOfCols * defaultFigSize[0], nrOfRows * defaultFigSize[1])
        fig = plt.figure(figsize=figsize)
        figAxesParameterDict = {"fig": fig, "nrOfRows": nrOfRows, "nrOfCols": nrOfCols, "currentAxes": 0}
    for i, entryIdentifier in enumerate(allEntryIdentifiersPlusFolderContents):
        if figAxesParameterDict:
            figAxesParameterDict["currentAxes"] = i + 1
        FolderContentPatchPlotter().plotPatchesOf(entryIdentifier, orderedJunctionsPerCellFilenameKey, measureDataFilenameKey,
                      allAxesParameter=allAxesParameter, selectedSubMeasure=selectedSubMeasure, colorBarLabel=colorBarLabel,
                      figAxesParameterDict=figAxesParameterDict, setAxesParameterInSinglePlots=setAxesParameterInSinglePlots,
                      genotypeToScenarioName=genotypeToScenarioName, timePointToName=timePointToName)
    if figAxesParameterDict:
        plt.show()

def calcOffsetToAlignPlyFileWithContours(plyFilename, entryIdentifier, surfaceContourPerCellFilenameKey="cellContours",
                                         labelColName="label", positionColumns=["x", "y", "z"]):
    from pyntcloud import PyntCloud
    pointCloud = PyntCloud.from_file(plyFilename)
    points = pointCloud.points
    allFolderContentsFilename = entryIdentifier[-1]
    multiFolderContent = MultiFolderContent(allFolderContentsFilename)
    folderContent = multiFolderContent.GetFolderContentOfIdentifier(entryIdentifier[:3])
    outlinesOfCells = folderContent.LoadKeyUsingFilenameDict(surfaceContourPerCellFilenameKey, convertDictKeysToInt=True, convertDictValuesToNpArray=True)
    offsets = []
    for cellLabel, currentOutline in outlinesOfCells.items():
        isLabel = points[labelColName] == cellLabel
        pointsOfCell = points.loc[isLabel, positionColumns].to_numpy()
        contourMean = np.mean(currentOutline, axis=0)
        pointsMean = np.mean(pointsOfCell, axis=0)
        offsets.append(contourMean - pointsMean)
    meanOffset = np.mean(offsets, axis=0)
    return meanOffset

def mainFig2AB(save=False, resultsFolder="Results/Tissue Visualization/", zoomedIn=False,
               measureDataFilenameKey=None, selectedSubMeasure=None, selectedSubMeasureName=None, colorMapValueRange=None):
    baseFilename = f"{resultsFolder}{'{}'}/figure2AB_{'{}'}_{'zoomedIn' if zoomedIn else 'overview'}_patches.png"
    if not measureDataFilenameKey is None:
        if selectedSubMeasureName is None:
            selectedSubMeasureName = selectedSubMeasure
        baseFilename = baseFilename.replace(".png", f"_{selectedSubMeasureName}.png")
        surfaceContourPerCellFilenameKey = "orderedJunctionsPerCellFilename"
        overlaidContourEdgePerCellFilenameKey = None
    else:
        surfaceContourPerCellFilenameKey = "cellContours"
        overlaidContourEdgePerCellFilenameKey = "orderedJunctionsPerCellFilename"
    allEntryIdentifiersPlusFolderContents = [
        ["WT inflorescence meristem", "P2", "T0", "Images/Matz2022SAM/Matz2022SAM.pkl"],
        ["WT", "20200220 WT S1", "120h", "Images/full cotyledons/full cotyledons.pkl"],
    ]
    selectedCellIds = {("WT inflorescence meristem", "P2"): [398, 381, 379, 417, 431],
                       ("WT", "20200220 WT S1"): [841860, 841670, 841855, 841669, 841666]}
    parameter = dict(surfaceContourPerCellFilenameKey=surfaceContourPerCellFilenameKey,
                     overlaidContourEdgePerCellFilenameKey=overlaidContourEdgePerCellFilenameKey,
                     measureDataFilenameKey=measureDataFilenameKey,
                     selectedSubMeasure=selectedSubMeasure,
                     setAxesParameterInSinglePlots=False,#not save, # to redo scaling of tissue set to True and allAxesParameter to {}
                     showTitle=False,
                     )
    scaleBarSize = 10
    if zoomedIn:
        allAxesParameter = {"WT inflorescence meristem_P2_T0": {'xlim': [34.59, 44.78], 'ylim': [48.72, 60.58], 'zlim': [17.46, 20.58], 'azim': 120, 'elev': 50}, #{'xlim': [35.32, 44.48], 'ylim': [48.78, 59.45], 'zlim': [17.75, 20.55], 'azim': 120, 'elev': 50},
                            "WT_20200220 WT S1_120h": {'xlim': [319.47, 370.72], 'ylim': [502.51, 554.61], 'zlim': [1.07, 3.93], 'azim': -90, 'elev': 90},}
        allScaleBarOffsets = {("WT inflorescence meristem", "P2"): np.array([32, 58, 17.7]),
                              ("WT", "20200220 WT S1"): np.array([380, 495, 0])}
        offsetOfOverlaidContours = {}
        offsetOfContoursOfTissues = {("WT inflorescence meristem", "P2"): [2.0914, 0.7689999999999984, 1.2651999999999965],
                                     ("WT", "20200220 WT S1"): [7.08, 6.2]}
    else:
        allAxesParameter = {"WT inflorescence meristem_P2_T0": {'xlim': [19.55, 76.43], 'ylim': [21.73, 73.37], 'zlim': [5.88, 18.46], 'azim': 130.0, 'elev': 86.8},
                            "WT_20200220 WT S1_120h": {'xlim': [124.22, 423.17], 'ylim': [174.65, 576.6], 'zlim': [1.05, 3.95], 'azim': -90, 'elev': 90}, }
        allScaleBarOffsets = {("WT inflorescence meristem", "P2"): np.array([5, 68, 0]),
                              ("WT", "20200220 WT S1"): np.array([450, 50, 0])}
        offsetOfOverlaidContours = {("WT inflorescence meristem", "P2"): [2.0914, 0.7689999999999984, 1.2651999999999965],
                                     ("WT", "20200220 WT S1"): [7.08, 6.2]}
        offsetOfContoursOfTissues = {}
    if not measureDataFilenameKey is None:
        if colorMapValueRange is None:
            colorMapValueRange = calcColorMapValueRange(allEntryIdentifiersPlusFolderContents, parameter, selectedCellIds)
        colorMapper = PatchCreator().createColorMapper(colorMapValueRange=colorMapValueRange, alphaAsFloat=0.5)
        parameter["colorMapper"] = colorMapper
    for entryIdentifier in allEntryIdentifiersPlusFolderContents:
        scenarioReplicateId = tuple(entryIdentifier[:2])
        if zoomedIn and scenarioReplicateId in selectedCellIds:
            currentSelectedCellIds = selectedCellIds[scenarioReplicateId]
        else:
            currentSelectedCellIds = None
        if scenarioReplicateId in offsetOfOverlaidContours:
            overlaidContourOffset = offsetOfOverlaidContours[scenarioReplicateId]
        else:
            overlaidContourOffset = None
        if scenarioReplicateId in offsetOfContoursOfTissues:
            offsetOfContours = offsetOfContoursOfTissues[scenarioReplicateId]
        else:
            offsetOfContours = None
        if scenarioReplicateId in allScaleBarOffsets:
            scaleBarOffset = allScaleBarOffsets[scenarioReplicateId]
        else:
            scaleBarOffset = None
        if scenarioReplicateId == ("WT inflorescence meristem", "P2"):
            backgroundFilename = "Images/Matz2022SAM/WT inflorescence meristem/P2/T0/20190416 myr-yfp T0 P2 mesh3_full outlines other.ply"
            loadBackgroundFromKey = None
            is3DBackground = True
            backgroundImageOffset = np.array([0, 0, -0.25])
            offsetToAlignPointsToOutline = calcOffsetToAlignPlyFileWithContours(backgroundFilename, entryIdentifier)
            backgroundImageOffset += offsetToAlignPointsToOutline
        else:
            backgroundFilename = "Images/Matz2022SAM/WT/20200220 WT S1/20200220 WT S1_reduced_test(3)_reexported with signal.ply"
            backgroundFilename = "Images/Matz2022SAM/WT/20200220 WT S1/20200220 WT S1_reduced_test(2)_reexport with signal.ply"
            loadBackgroundFromKey = None # "originalImageFilename"
            is3DBackground = True
            backgroundImageOffset = np.array([471.5/2, 315/2, -0.25])
        backgroundFilename = None
        ax = FolderContentPatchPlotter().plotPatchesOf(entryIdentifier, allAxesParameter=allAxesParameter, overlaidContourOffset=overlaidContourOffset, offsetOfContours=offsetOfContours,
                      genotypeToScenarioName=genotypeToScenarioName, timePointToName=timePointToName, plotOutlines=True,
                      # overlaidContourDistancesColorMapper=overlaidContourDistancesColorMapper,
                      selectedCellLabels=currentSelectedCellIds, convertFromIdToLabel=False, scaleBarSize=scaleBarSize, scaleBarOffset=scaleBarOffset,
                      backgroundFilename=backgroundFilename, loadBackgroundFromKey=loadBackgroundFromKey, is3DBackground=is3DBackground, backgroundImageOffset=backgroundImageOffset,
                      **parameter)
        if save:
            scenarioName = Path(entryIdentifier[-1]).stem
            saveAsFilename = baseFilename.format(scenarioName, scenarioReplicateId[0])
            Path(saveAsFilename).parent.mkdir(parents=True, exist_ok=True)
            print(saveAsFilename)
            plt.savefig(saveAsFilename, bbox_inches="tight", dpi=300)
            plt.close()
        else:
            plt.show()

        if parameter["setAxesParameterInSinglePlots"]:
            decimal = 2
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            zlim = ax.get_zlim()
            azim = ax.azim
            elev = ax.elev
            axesParameter = {"xlim": np.round(xlim, decimal).tolist(), "ylim": np.round(ylim, decimal).tolist(), "zlim": np.round(zlim, decimal).tolist(), "azim": np.round(azim, decimal), "elev": np.round(elev, decimal)}
            print(f'{axesParameter}')

def convertCellLabelsToId(folderContent: FolderContent, cellLabels: list, contourFilenameKey: str = "contourFilename", printOut: bool = True):
    myCellIdTracker = CellIdTracker()
    myCellIdTracker.RunCellIdTracker(contourFile=folderContent.GetFilenameDictKeyValue(contourFilenameKey))
    labelsToIdConverter = myCellIdTracker.GetLabelsToIdsDict()
    labelIds = [labelsToIdConverter[label] for label in cellLabels]
    if printOut:
        print(labelIds)
    return labelIds

def calcColorMapValueRange(allEntryIdentifiersPlusFolderContents, parameter, selectedCellIds=None):
    values = []
    for entryIdentifier in allEntryIdentifiersPlusFolderContents:
        allFolderContentsFilename = entryIdentifier[-1]
        multiFolderContent = MultiFolderContent(allFolderContentsFilename)
        folderContent = multiFolderContent.GetFolderContentOfIdentifier(entryIdentifier[:3])
        contours = folderContent.LoadKeyUsingFilenameDict(parameter["surfaceContourPerCellFilenameKey"], convertDictKeysToInt=True, convertDictValuesToNpArray=True)
        cellLabelsOfContours = list(contours.keys())
        measureDataFilenameKey = parameter["measureDataFilenameKey"]
        selectedSubMeasure = parameter["selectedSubMeasure"]
        valuesOfCells = folderContent.LoadKeyUsingFilenameDict(measureDataFilenameKey)
        if type(selectedSubMeasure) != dict:
            valuesOfCells = valuesOfCells[selectedSubMeasure]
        else:
            valuesOfCells = FolderContentPatchPlotter().calculateRatioMeasureData(valuesOfCells, selectedSubMeasure)
        if not selectedCellIds is None:
            selectedCellLabels = selectedCellIds[(entryIdentifier[0], entryIdentifier[1])]
            for cellLabel in cellLabelsOfContours:
                if not cellLabel in selectedCellLabels:
                    valuesOfCells.pop(cellLabel)
        values.extend(list(valuesOfCells.values()))
    colorMapValueRange = [np.min(values), np.max(values)]
    return colorMapValueRange

def mainFig3AOrB(save=True, showAllTogether=True, figA=False, resultsFolder="Results/Methodology Visualization/3A/",
                 selectedSubMeasureOfB="angleGiniCoeff", measureDataFilenameKey="regularityMeasuresFilename",
                 selectedSubMeasureName=None, transposeFigureOutline=False, colorMapValueRange=None):
    sys.path.insert(0, "./Images/")
    from InputData import GetResolutions
    genotypesResolutionDict = GetResolutions()
    # Ryan's tissue selections uses col-0_20170327 WT S1, 20180618 ktn1-2 S1
    replicateNamesOfGenotypes = {"col-0": "20170501 WT S1",
                                 "Oryzalin": "20170731 oryzalin S2",
                                 "ktn1-2":"20180618 ktn1-2 S2"}
    selectedCellIds = {("col-0", "20170501 WT S1"): [9, 6, 5, 7],
                       ("Oryzalin", "20170731 oryzalin S2"): [9, 11, 10, 6, 5],
                       ("ktn1-2", "20180618 ktn1-2 S2"): [27, 29, 28, 6]}
    if figA:
        saveAsFilename = "{}figure_patches.png"
        allTimePoints = ["0h", "24h", "48h", "72h", "96h"]
        parameter = dict(surfaceContourPerCellFilenameKey="cellContours",
                         overlaidContourEdgePerCellFilenameKey="orderedJunctionsPerCellFilename",
                         measureDataFilenameKey=None,
                         getCellIdByKeyStroke=False, # to redo scaling of tissue set to True and allAxesParameter to {}
                         scaleBarSize=20,
                         genotypesResolutionDict=genotypesResolutionDict,
                         isThreeDimensional=False,
                         )
    else:
        onlyTimePoint = "96h"
        if selectedSubMeasureName is None:
            selectedSubMeasureName = selectedSubMeasureOfB
        saveAsFilename = "{}figure_patchesOf_{}_{}.png".format("{}", selectedSubMeasureName, onlyTimePoint)
        allTimePoints = [onlyTimePoint]
        parameter = dict(surfaceContourPerCellFilenameKey="orderedJunctionsPerCellFilename",
                         measureDataFilenameKey=measureDataFilenameKey,
                         selectedSubMeasure=selectedSubMeasureOfB,
                         showColorBar=False,
                         getCellIdByKeyStroke=False,  # to redo scaling of tissue set to True and allAxesParameter to {}
                         scaleBarSize=20,
                         genotypesResolutionDict=genotypesResolutionDict,
                         isThreeDimensional=False,
                         )
    parameter["patchKwargs"] = dict(outlineLineWidth=4, polygonOutlineLineWidth= 2)
    allEntryIdentifiersPlusFolderContents = []
    for scenarioName, replicateName in replicateNamesOfGenotypes.items():
        if type(replicateName) == str:
            for t in allTimePoints:
                allEntryIdentifiersPlusFolderContents.append([scenarioName, replicateName, t, "Images/Eng2021Cotyledons/Eng2021Cotyledons.pkl"])
        else:
            for r in replicateName:
                for t in allTimePoints:
                    allEntryIdentifiersPlusFolderContents.append([scenarioName, r, t, "Images/Eng2021Cotyledons/Eng2021Cotyledons.pkl"])
    saveAsFilename = saveAsFilename.format(resultsFolder)
    allAxesParameter = {}
    if showAllTogether:
        nrOfCols = len(allTimePoints)
        nrOfRows = len(allEntryIdentifiersPlusFolderContents) // nrOfCols
        if transposeFigureOutline:
            nrOfCols, nrOfRows = nrOfRows, nrOfCols
            nrOfAxes = nrOfCols * nrOfRows # not -1 as axes start with index of 1
        if not figA:
            if colorMapValueRange is None:
                colorMapValueRange = calcColorMapValueRange(allEntryIdentifiersPlusFolderContents, parameter, selectedCellIds)
            colorMapper = PatchCreator().createColorMapper(colorMapValueRange=colorMapValueRange, alphaAsFloat=0.5)
            parameter["colorMapper"] = colorMapper
            plt.colorbar(parameter["colorMapper"])
            if not saveAsFilename is None:
                saveColorBarAsFilename = Path(saveAsFilename).with_name(Path(saveAsFilename).stem + "_colorBar.png")
                Path(saveColorBarAsFilename).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(saveColorBarAsFilename, bbox_inches="tight", dpi=300)
                plt.close()
        baseSizeOfAxisInInch = 4
        figSize = (nrOfCols * baseSizeOfAxisInInch, nrOfRows * baseSizeOfAxisInInch)
        fig = plt.figure(figsize=figSize)
        figAxesParameterDict = {"fig": fig, "nrOfRows": nrOfRows, "nrOfCols": nrOfCols, "currentAxes": 1}
    else:
        figAxesParameterDict = None
    for i, entryIdentifier in enumerate(allEntryIdentifiersPlusFolderContents):
        scenarioReplicateId = (entryIdentifier[0], entryIdentifier[1])
        if scenarioReplicateId in selectedCellIds:
            currentSelectedCellIds = selectedCellIds[scenarioReplicateId]
        else:
            currentSelectedCellIds = None
        FolderContentPatchPlotter().plotPatchesOf(entryIdentifier, allAxesParameter=allAxesParameter,
                                                  genotypeToScenarioName=genotypeToScenarioName, timePointToName=timePointToName,
                                                  selectedCellLabels=currentSelectedCellIds, figAxesParameterDict=figAxesParameterDict,
                                                  **parameter)
        if not showAllTogether:
            plt.show()
        else:
            if transposeFigureOutline:
                figAxesParameterDict["currentAxes"] += nrOfCols
                if figAxesParameterDict["currentAxes"] > nrOfAxes:
                    figAxesParameterDict["currentAxes"] -= nrOfAxes - 1 # going one column further
            else:
                figAxesParameterDict["currentAxes"] += 1
    if save:
        if not saveAsFilename is None:
            Path(saveAsFilename).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(saveAsFilename, bbox_inches="tight", dpi=300)
            plt.close()
    else:
        if showAllTogether:
            plt.show()

def determineValueRangeForMultipleSubMeasures(measureDataFilenameKey, selectedSubMeasures, allTimePoints = ["96h"]):
    replicateNamesOfGenotypes = {"col-0": "20170501 WT S1",
                                 "Oryzalin": "20170731 oryzalin S2",
                                 "ktn1-2":"20180618 ktn1-2 S2",}
    selectedCellIds = {("col-0", "20170501 WT S1"): [9, 6, 5, 7],
                       ("Oryzalin", "20170731 oryzalin S2"): [9, 11, 10, 6, 5],
                       ("ktn1-2", "20180618 ktn1-2 S2"): [27, 29, 28, 6]}
    allEntryIdentifiersPlusFolderContents = []
    for scenarioName, replicateName in replicateNamesOfGenotypes.items():
        if type(replicateName) == str:
            for t in allTimePoints:
                allEntryIdentifiersPlusFolderContents.append([scenarioName, replicateName, t, "Images/Eng2021Cotyledons/Eng2021Cotyledons.pkl"])
        else:
            for r in replicateName:
                for t in allTimePoints:
                    allEntryIdentifiersPlusFolderContents.append([scenarioName, r, t, "Images/Eng2021Cotyledons/Eng2021Cotyledons.pkl"])
    colorMapValueRange = []
    for subMeasureKey in selectedSubMeasures:
        parameter = dict(surfaceContourPerCellFilenameKey="orderedJunctionsPerCellFilename", measureDataFilenameKey=measureDataFilenameKey, selectedSubMeasure=subMeasureKey)
        currentRange = calcColorMapValueRange(allEntryIdentifiersPlusFolderContents, parameter, selectedCellIds)
        colorMapValueRange.extend(currentRange)
    return [np.min(colorMapValueRange), np.max(colorMapValueRange)]

def mainSupFig1A(save=False, resultsFolder="Results/Tissue Visualization/", zoomedIn=False,
               measureDataFilenameKey=None, selectedSubMeasure=None, selectedSubMeasureName=None, colorMapValueRange=None):
    baseFilename = f"{resultsFolder}{'{}'}/supFigure1A_{'{}'}_{'zoomedIn' if zoomedIn else 'overview'}_patches.png"
    if not measureDataFilenameKey is None:
        if selectedSubMeasureName is None:
            selectedSubMeasureName = selectedSubMeasure
        baseFilename = baseFilename.replace(".png", f"_{selectedSubMeasureName}.png")
        surfaceContourPerCellFilenameKey = "orderedJunctionsPerCellFilename"
        overlaidContourEdgePerCellFilenameKey = None
    else:
        surfaceContourPerCellFilenameKey = "cellContours"
        overlaidContourEdgePerCellFilenameKey = "orderedJunctionsPerCellFilename"
    allEntryIdentifiersPlusFolderContents = [
        # ["speechless", "20210712_R1M001A", "96h", "Images/full cotyledons/full cotyledons speechless.pkl"],
        # ["speechless", "20210712_R2M001A", "96h", "Images/full cotyledons/full cotyledons speechless.pkl"],
        ["speechless", "20210712_R5M001", "96h", "Images/full cotyledons/full cotyledons speechless.pkl"],
    ]
    selectedCellIds = {("speechless", "20210712_R5M001"): [543, 357, 1155, 945]}
    parameter = dict(surfaceContourPerCellFilenameKey=surfaceContourPerCellFilenameKey,
                     overlaidContourEdgePerCellFilenameKey=overlaidContourEdgePerCellFilenameKey,
                     measureDataFilenameKey=measureDataFilenameKey,
                     selectedSubMeasure=selectedSubMeasure,
                     setAxesParameterInSinglePlots=False,#not save, # to redo scaling of tissue set to True and allAxesParameter to {}
                     showTitle=False,
                     getCellIdByKeyStroke=False,
                     showPlot=not save,
                     removeNotSelectedContourCells=False
                     )
    scaleBarSize = 10
    offsetOfContoursOfTissues = {("speechless", "20210712_R5M001"):[-24.5, -9.5]}
    offsetOfOverlaidContours = {}
    if zoomedIn:
        allAxesParameter = {"speechless_20210712_R5M001_96h": {'xlim': [477.58, 568.04], 'ylim': [378.11, 470.81], 'zlim': [1.27, 3.73], 'azim': -90, 'elev': 90}}
        allScaleBarOffsets = {("speechless", "20210712_R5M001"): [453, 364, 1.1]}
    else:
        allAxesParameter = {"speechless_20210712_R5M001_96h": {'xlim': [157.28, 586.98], 'ylim': [225.89, 735.06], 'zlim': [1.11, 3.89], 'azim': -90, 'elev': 90}}
        allScaleBarOffsets = {("speechless", "20210712_R5M001"): [639, 116, 1.1]}
    if not measureDataFilenameKey is None:
        if colorMapValueRange is None:
            colorMapValueRange = calcColorMapValueRange(allEntryIdentifiersPlusFolderContents, parameter, selectedCellIds)
        colorMapper = PatchCreator().createColorMapper(colorMapValueRange=colorMapValueRange, alphaAsFloat=0.5)
        parameter["colorMapper"] = colorMapper
    for entryIdentifier in allEntryIdentifiersPlusFolderContents:
        scenarioReplicateId = tuple(entryIdentifier[:2])
        currentSelectedCellIds = None
        if zoomedIn and scenarioReplicateId in selectedCellIds:
            currentSelectedCellIds = selectedCellIds[scenarioReplicateId]
        if scenarioReplicateId in offsetOfOverlaidContours:
            overlaidContourOffset = offsetOfOverlaidContours[scenarioReplicateId]
        else:
            overlaidContourOffset = None
        if scenarioReplicateId in offsetOfContoursOfTissues:
            offsetOfContours = offsetOfContoursOfTissues[scenarioReplicateId]
        else:
            offsetOfContours = None
        if scenarioReplicateId in allScaleBarOffsets:
            scaleBarOffset = allScaleBarOffsets[scenarioReplicateId]
        else:
            scaleBarOffset = None
        backgroundImageOffset = None
        loadBackgroundFromKey = None
        is3DBackground = False
        backgroundFilename = None
        ax = FolderContentPatchPlotter().plotPatchesOf(entryIdentifier, figAxesParameterDict=None, allAxesParameter=allAxesParameter, overlaidContourOffset=overlaidContourOffset, offsetOfContours=offsetOfContours,
                                                       genotypeToScenarioName=genotypeToScenarioName, timePointToName=timePointToName, plotOutlines=True,
                                                       # overlaidContourDistancesColorMapper=overlaidContourDistancesColorMapper,
                                                       selectedCellLabels=currentSelectedCellIds, convertFromIdToLabel=False, scaleBarSize=scaleBarSize, scaleBarOffset=scaleBarOffset,
                                                       backgroundFilename=backgroundFilename, loadBackgroundFromKey=loadBackgroundFromKey, is3DBackground=is3DBackground, backgroundImageOffset=backgroundImageOffset,
                                                       **parameter)
        if save:
            scenarioName = Path(entryIdentifier[-1]).stem
            saveAsFilename = baseFilename.format(scenarioName, scenarioReplicateId[0])
            Path(saveAsFilename).parent.mkdir(parents=True, exist_ok=True)
            print(saveAsFilename)
            plt.savefig(saveAsFilename, bbox_inches="tight", dpi=300)
            plt.close()

        if parameter["setAxesParameterInSinglePlots"]:
            decimal = 2
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            zlim = ax.get_zlim()
            azim = ax.azim
            elev = ax.elev
            axesParameter = {"xlim": np.round(xlim, decimal).tolist(), "ylim": np.round(ylim, decimal).tolist(), "zlim": np.round(zlim, decimal).tolist(), "azim": np.round(azim, decimal), "elev": np.round(elev, decimal)}
            print(f'{axesParameter}')

if __name__ == '__main__':
    # mainFig2AB(save=True, zoomedIn=True)
    # mainFig2AB(save=True, zoomedIn=False)
    # mainFig2AB(save=True, zoomedIn=False)
    # mainFig2AB(save=True, zoomedIn=True, measureDataFilenameKey="regularityMeasuresFilename", selectedSubMeasure="lengthGiniCoeff")
    # mainFig2AB(save=True, zoomedIn=True, measureDataFilenameKey="regularityMeasuresFilename", selectedSubMeasure="angleGiniCoeff")
    # mainFig2AB(save=True, zoomedIn=True, colorMapValueRange=[0.9276042480508213, 1.2338458881519452],measureDataFilenameKey="areaMeasuresPerCell",
    #            selectedSubMeasure={"ratio": ["regularPolygonArea", "labelledImageArea"]}, selectedSubMeasureName="ratio_regularPolygonArea_labelledImageArea")
    # mainFig2AB(save=True, zoomedIn=True, colorMapValueRange=[0.9276042480508213, 1.2338458881519452], measureDataFilenameKey="areaMeasuresPerCell",
    #            selectedSubMeasure={"ratio": ["originalPolygonArea", "labelledImageArea"]}, selectedSubMeasureName="ratio_originalPolygonArea_labelledImageArea")
    # mainFig3AOrB(save=False, figA=True, transposeFigureOutline=True)
    # mainFig3AOrB(save=True, figA=False, selectedSubMeasureOfB="angleGiniCoeff")
    # mainFig3AOrB(save=True, figA=False, selectedSubMeasureOfB="lengthGiniCoeff")
    # combinedAreaRatioValueRange = determineValueRangeForMultipleSubMeasures("areaMeasuresPerCell", [{"ratio": ["originalPolygonArea", "labelledImageArea"]}, {"ratio": ["regularPolygonArea", "labelledImageArea"]}])
    # mainFig3AOrB(save=True, figA=False, resultsFolder="Results/Methodology Visualization/5/", transposeFigureOutline=True,
    #              measureDataFilenameKey="areaMeasuresPerCell", selectedSubMeasureOfB={"ratio": ["originalPolygonArea", "labelledImageArea"]},
    #              selectedSubMeasureName="ratio_originalPolygonArea_labelledImageArea",
    #              colorMapValueRange=combinedAreaRatioValueRange)
    # mainFig3AOrB(save=True, figA=False, resultsFolder="Results/Methodology Visualization/5/", transposeFigureOutline=True,
    #              measureDataFilenameKey="areaMeasuresPerCell", selectedSubMeasureOfB={"ratio": ["regularPolygonArea", "labelledImageArea"]},
    #              selectedSubMeasureName="ratio_regularPolygonArea_labelledImageArea",
    #              colorMapValueRange=combinedAreaRatioValueRange)
    # mainSupFig1A(save=True, zoomedIn=False)
    mainSupFig1A(save=False, zoomedIn=True, measureDataFilenameKey="regularityMeasuresFilename", selectedSubMeasure="lengthGiniCoeff", colorMapValueRange=[0.011224633401410082, 0.1875771799706416])
    # mainSupFig1A(save=True, zoomedIn=True, measureDataFilenameKey="regularityMeasuresFilename", selectedSubMeasure="angleGiniCoeff", colorMapValueRange=[0.00931882037513563, 0.08732278219992405])