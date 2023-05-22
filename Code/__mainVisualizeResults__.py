import sys

sys.path.insert(0, "./Code/Analyze/")
sys.path.insert(0, "./Code/Analyze/PlacingValuesOnSkales/")

from ExtendedPlotter import plotPatchyCotyledonResults
from FolderContentPatchPlotter import mainFig2AB, mainFig3AOrB, determineValueRangeForMultipleSubMeasures
from PolygonRegularityPlotterAlongAxis import visualizeCellsAlongAxisMain, visualizeArtificialPolygonsAlongTwoAxisMain

def createAllPythonRelatedSubfigures():
    # Fig. 1 A
    visualizeCellsAlongAxisMain()
    # Fig. 1 B-D
    visualizeArtificialPolygonsAlongTwoAxisMain()
    # Fig. 2 A, B
    mainFig2AB(save=True, zoomedIn=False, resultsFolder="Results/Tissue Visualization/")
    mainFig2AB(save=True, zoomedIn=True, measureDataFilenameKey="regularityMeasuresFilename", selectedSubMeasure="lengthGiniCoeff", resultsFolder="Results/Tissue Visualization/")
    mainFig2AB(save=True, zoomedIn=True, measureDataFilenameKey="regularityMeasuresFilename", selectedSubMeasure="angleGiniCoeff", resultsFolder="Results/Tissue Visualization/")
    # Fig. 2 C
    plotPatchyCotyledonResults(plotRegularityMeasures=True, plotCombinedDataSet=True)
    # Fig. 3 A, B
    irregularityOfEng2021CotyledonsFolder = "Results/Tissue Visualization/Eng2021Cotyledons/Fig3/"
    mainFig3AOrB(save=True, figA=True, transposeFigureOutline=True, resultsFolder=irregularityOfEng2021CotyledonsFolder)
    mainFig3AOrB(save=True, figA=False, selectedSubMeasureOfB="lengthGiniCoeff", resultsFolder=irregularityOfEng2021CotyledonsFolder)
    mainFig3AOrB(save=True, figA=False, selectedSubMeasureOfB="angleGiniCoeff", resultsFolder=irregularityOfEng2021CotyledonsFolder)
    # Fig. 3 C, D
    plotPatchyCotyledonResults(plotRegularityMeasures=True, plotFullDataSet=True)
    # Fig. 4 B, C
    mainFig2AB(save=True, zoomedIn=True, colorMapValueRange=[0.9276042480508213, 1.2338458881519452],measureDataFilenameKey="areaMeasuresPerCell",
               selectedSubMeasure={"ratio": ["regularPolygonArea", "labelledImageArea"]}, selectedSubMeasureName="ratio_regularPolygonArea_labelledImageArea")
    mainFig2AB(save=True, zoomedIn=True, colorMapValueRange=[0.9276042480508213, 1.2338458881519452], measureDataFilenameKey="areaMeasuresPerCell",
               selectedSubMeasure={"ratio": ["originalPolygonArea", "labelledImageArea"]}, selectedSubMeasureName="ratio_originalPolygonArea_labelledImageArea")
    # Fig. 4 D
    plotPatchyCotyledonResults(plotAreaMeasures=True, plotCombinedDataSet=True)
    # Fig. 5 A
    irregularityOfEng2021CotyledonsFolder = "Results/Tissue Visualization/Eng2021Cotyledons/Fig5/"
    combinedAreaRatioValueRange = determineValueRangeForMultipleSubMeasures("areaMeasuresPerCell", [{"ratio": ["originalPolygonArea", "labelledImageArea"]}, {"ratio": ["regularPolygonArea", "labelledImageArea"]}])
    mainFig3AOrB(save=True, figA=False, resultsFolder=irregularityOfEng2021CotyledonsFolder, transposeFigureOutline=True,
                 measureDataFilenameKey="areaMeasuresPerCell", selectedSubMeasureOfB={"ratio": ["originalPolygonArea", "labelledImageArea"]},
                 selectedSubMeasureName="ratio_originalPolygonArea_labelledImageArea",
                 colorMapValueRange=combinedAreaRatioValueRange)
    mainFig3AOrB(save=True, figA=False, resultsFolder=irregularityOfEng2021CotyledonsFolder, transposeFigureOutline=True,
                 measureDataFilenameKey="areaMeasuresPerCell", selectedSubMeasureOfB={"ratio": ["regularPolygonArea", "labelledImageArea"]},
                 selectedSubMeasureName="ratio_regularPolygonArea_labelledImageArea",
                 colorMapValueRange=combinedAreaRatioValueRange)
    # Fig. 5 B, C
    plotPatchyCotyledonResults(plotAreaMeasures=True, plotFullDataSet=True)

if __name__ == '__main__':
    createAllPythonRelatedSubfigures()