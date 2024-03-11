import itertools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np
import pandas as pd
import seaborn as sns

from BasePlotter import BasePlotter
from copy import deepcopy
from ExtendedPlotter import ExtendedPlotter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class PCAAnalysis(BasePlotter, ExtendedPlotter):

    table: pd.DataFrame = None
    selectedTable: pd.DataFrame = None
    scaledTable: pd.DataFrame = None
    loadings: pd.DataFrame = None
    pcScores: pd.DataFrame = None
    scaler: StandardScaler = None
    pc: PCA = None

    def __init__(self, table: pd.DataFrame or str, tableLoadKwargs: dict = {}):
        self.SetTable(table, tableLoadKwargs=tableLoadKwargs)

    def SetTable(self, table, tableLoadKwargs: dict = {}):
        if not isinstance(table, pd.DataFrame):
            self.table = pd.read_csv(table, **tableLoadKwargs)
        else:
            self.table = table

    def FilterTable(self, columnsExpectedValuesToKeep: dict, table: pd.DataFrame = None):
        if table is None:
            table = self.table
        isExpectedValue = np.ones(len(table), dtype=bool)
        for column, expectedValue in columnsExpectedValuesToKeep.items():
            isCurrentValueExpected = np.isin(table[column], expectedValue)
            isExpectedValue = isExpectedValue & isCurrentValueExpected
        shouldRowBeDropped = np.invert(isExpectedValue)
        if np.any(shouldRowBeDropped):
            rowsToDrop = np.where(shouldRowBeDropped)[0]
            table.drop(rowsToDrop, axis=0, inplace=True)

    def AnalyseTable(self, columnsToAnalyse: list or np.ndarray, numberOfComponents: int = 2, componentBaseName: str = "PC ",
                     columnsExpectedValuesToKeep: dict = None):
        self.scaler = StandardScaler()
        self.checkColumnPresence(columnsToAnalyse)
        self.selectedTable = deepcopy(self.table)
        if columnsExpectedValuesToKeep is not None:
            self.FilterTable(columnsExpectedValuesToKeep, self.selectedTable)
        self.selectedTable = self.selectedTable[columnsToAnalyse]
        self.scaler.fit(self.selectedTable)
        self.scaledTable = self.scaler.transform(self.selectedTable)
        self.pc = PCA(n_components=numberOfComponents)
        pcaColumnNames = [componentBaseName + str(i+1) for i in range(numberOfComponents)]
        self.pcScores = pd.DataFrame(self.pc.fit_transform(self.scaledTable), columns=pcaColumnNames)
        featureNames = self.selectedTable.columns
        self.loadings = pd.DataFrame(self.pc.components_.T, columns=pcaColumnNames, index=featureNames)

    def PlotBiPlot(self, ax: matplotlib.axes.Axes = None, fontSize: int = 20, pcXIdx: int = 0, pcYIdx: int = 1, title: str = None,
                   loadColorPalette: list = sns.color_palette("colorblind"), showFeatureLoadText: bool = False,
                   showScatterLabels: bool = False, showXLabel: bool = True, showYLabel: bool = True,
                   showVarianceOnLabel: bool = True, scatterColor: list or str = "black",
                   saveOrShowKwargs: dict = {"filenameToSave": None, "showPlot": True, "dpi": 300}):
        PC_X = self.pc.fit_transform(self.scaledTable)[:, pcXIdx]
        PC_Y = self.pc.fit_transform(self.scaledTable)[:, pcYIdx]
        componentLoadingValues = self.pc.components_
        scalePC_X = 1.0 / (PC_X.max() - PC_X.min())
        scalePC_Y = 1.0 / (PC_Y.max() - PC_Y.min())
        self.SetRcParams(fontSize=fontSize)
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 9), constrained_layout=True)
        else:
            fig = None

        featureNames = self.selectedTable.columns
        for i, feature in enumerate(featureNames):
            loadColor = loadColorPalette[i]
            ax.arrow(0, 0, componentLoadingValues[pcXIdx, i], componentLoadingValues[pcYIdx, i],
                     head_width=0.03, head_length=0.03, color=loadColor)
            if showFeatureLoadText:
                ax.text(componentLoadingValues[pcXIdx, i] * 1.15, componentLoadingValues[pcYIdx, i] * 1.15,
                        self.GetLabelOfMeasure(feature, requirePresenceInConverter=False),
                        color=loadColor, fontsize=fontSize-int(0.1 * fontSize))

        ax.scatter(PC_X * scalePC_X, PC_Y * scalePC_Y, s=5, color=scatterColor)

        if showScatterLabels:
            for i, label in enumerate(self.pcScores.index):
                ax.text(PC_X[i] * scalePC_X,
                        PC_Y[i] * scalePC_Y, str(label),
                        fontsize=fontSize//2)

        explainedVariancePercentage = 100 * self.pc.explained_variance_ratio_
        if showXLabel:
            xLabel = f"PC{pcXIdx+1}"
            if showVarianceOnLabel:
                xLabel += f" {explainedVariancePercentage[pcXIdx] : .2f}%"
            ax.set_xlabel(xLabel, fontsize="small")
        if showYLabel:
            yLabel = f"PC{pcYIdx+1}"
            if showVarianceOnLabel:
                yLabel += f" {explainedVariancePercentage[pcYIdx] : .2f}%"
            ax.set_ylabel(yLabel, fontsize="small")
        if title is not None:
            ax.set_title(title, fontsize="large")
        self.SaveOrShowFigure(**saveOrShowKwargs)
        return fig, ax

    def checkColumnPresence(self, columnsToAnalyse: list or np.ndarray):
        presentColumns = self.table.columns
        isColumnToAnalysePresent = np.isin(columnsToAnalyse, presentColumns)
        assert np.all(isColumnToAnalysePresent), f"The columns {np.array(columnsToAnalyse)[np.invert(isColumnToAnalysePresent)]} are not present in the table with the present columns {presentColumns.tolist()}"

def addLegend(labels, colors, legendKwargs: dict = {}):
    handles = []
    for label, color in zip(labels, colors):
        currentLegendHandle = lines.Line2D([0], [0], label=label, color=color)
        handles.append(currentLegendHandle)
    plt.legend(handles=handles, **legendKwargs)

def createLoadColorPalette():
    loadColorPalette: list = list(sns.color_palette("colorblind"))
    loadColorPalette = [loadColorPalette[1], loadColorPalette[3], loadColorPalette[0], loadColorPalette[2]]
    return loadColorPalette

def plotFilteredPCA(analyser, columnsToAnalyse, columnsExpectedValuesToKeep, i, ax, nrOfTimePoints, loadColorPalette, hideInnerAxisLabels, numberOfPlots, t,
                    colorColumnName: str = "color"):
    analyser.AnalyseTable(columnsToAnalyse, numberOfComponents=4, columnsExpectedValuesToKeep=columnsExpectedValuesToKeep)
    if i < nrOfTimePoints:
        title = f"{t}"
    else:
        title = None
    if hideInnerAxisLabels:
        if i > numberOfPlots - nrOfTimePoints - 1:
            showXLabel = True
        else:
            showXLabel = False
        if i % nrOfTimePoints == 0:
            showYLabel = True
        else:
            showYLabel = False
    else:
        showXLabel, showYLabel = True, True
    if colorColumnName is not None:
        scatterColor = analyser.table.iloc[analyser.selectedTable.index][colorColumnName].values
    else:
        scatterColor = None
    analyser.PlotBiPlot(ax=ax[i], pcXIdx=pcXIdx, pcYIdx=pcYIdx, loadColorPalette=loadColorPalette, title=title,
                        scatterColor=scatterColor,
                        saveOrShowKwargs={"showPlot": False}, showXLabel=showXLabel, showYLabel=showYLabel)
def plotIndividualGenotypeTimePointPCAs(analyser: PCAAnalysis, columnsToAnalyse: list,
                                        allTimePoints: list = ["0h", "24h", "48h", "72h", "96h"],
                                        genotypes: list = ["col-0", "Oryzalin", "ktn1-2"],
                                        hideInnerAxisLabels: bool = False):
    nrOfGenotypes = len(genotypes) if len(genotypes) > 0 else 1
    nrOfTimePoints = len(allTimePoints) if len(allTimePoints) > 0 else 1
    figsize = [5 * nrOfTimePoints, 5 * nrOfGenotypes]
    fig, ax = plt.subplots(nrOfGenotypes, nrOfTimePoints, figsize=figsize)

    if not hideInnerAxisLabels:
        fig.tight_layout()
        bottomSpacing = 0.1
        leftSpacing = bottomSpacing * figsize[1] / figsize[0]
        plt.subplots_adjust(left=leftSpacing, bottom=bottomSpacing, right=None, top=None, wspace=None, hspace=None)
    ax = ax.ravel()
    numberOfPlots = len(ax)
    loadColorPalette = createLoadColorPalette()
    if nrOfGenotypes > 1 and nrOfTimePoints > 1:
        for i, (g, t) in enumerate(itertools.product(genotypes, allTimePoints)):
            columnsExpectedValuesToKeep = {"genotype": g, "time point": t}
            plotFilteredPCA(analyser, columnsToAnalyse, columnsExpectedValuesToKeep, i, ax, nrOfTimePoints, loadColorPalette, hideInnerAxisLabels, numberOfPlots)
    elif nrOfGenotypes > 1:
        for i, g in enumerate(genotypes):
            columnsExpectedValuesToKeep = {"genotype": g}
            plotFilteredPCA(analyser, columnsToAnalyse, columnsExpectedValuesToKeep, i, ax, 0, loadColorPalette, hideInnerAxisLabels, numberOfPlots, t="None")
    else:
        for i, t in enumerate(allTimePoints):
            columnsExpectedValuesToKeep = {"time point": t}
            plotFilteredPCA(analyser, columnsToAnalyse, columnsExpectedValuesToKeep, i, ax, nrOfTimePoints, loadColorPalette, hideInnerAxisLabels, numberOfPlots, t=t)

    return fig, ax

def analyseEngCotyledonsRegularityIndividualPCA(pcXIdx: int = 0, pcYIdx: int = 1):
    tableFilename = "Results/combinedMeasures_Eng2021Cotyledons.csv"
    filenameToSave = f"Results/regularityResults/PCA/PCA_Eng2021Cotyledons{'' if pcXIdx == 0 and pcYIdx == 1 else f'_PC{pcXIdx+1}VS{pcYIdx+1}'}.png"
    columnsToAnalyse = ["angleGiniCoeff", "lengthGiniCoeff", "relativeCompleteness", "lobyness"]
    labelNameConverterDict = {"lengthGiniCoeff": "Gini coefficient of length", "angleGiniCoeff": "Gini coefficient of angle",
                              "relativeCompleteness": "relative completeness", "lobyness": "lobyness"}
    allTimePoints = ["0h", "24h", "48h", "72h", "96h"]
    genotypes = ["col-0", "Oryzalin", "ktn1-2"]
    genotypeNames = ["WT", "WT+Oryzalin", "$\it{ktn1}$-$\it{2}$"]

    analyser = PCAAnalysis(tableFilename)
    hideInnerAxisLabels = True
    fig, ax = plotIndividualGenotypeTimePointPCAs(analyser, columnsToAnalyse, allTimePoints=allTimePoints, genotypes=genotypes, hideInnerAxisLabels=hideInnerAxisLabels)
    if not hideInnerAxisLabels:
        xPos = 0
        genotypeYPositions = np.asarray([0.85, 0.55, 0.225])
    else:
        xPos = 0.07
        genotypeYPositions = [0.767, 0.5, 0.227]
    legendKwargs = {"loc": "lower center", "bbox_to_anchor": (0.5, 0), "bbox_transform": fig.transFigure, "ncol": 5}
    loadColorPalette = createLoadColorPalette()
    addLegend([labelNameConverterDict[label] for label in columnsToAnalyse], loadColorPalette, legendKwargs=legendKwargs)
    for g, yPos in zip(genotypeNames, genotypeYPositions):
        plt.gcf().text(xPos, yPos, g, fontsize="large", rotation="vertical", horizontalalignment="center", verticalalignment="center")
    saveOrShowKwargs = {"filenameToSave": None, "showPlot": True, "dpi": 300}
    analyser.SaveOrShowFigure(**saveOrShowKwargs)

def addGenotypeTimePointDependentColor(pcaAnalyser, genotypeColorConversion: dict, colorColumnName: str = "color"):
    timePointConversionFromStrToNumber = {"0h": 0, '12h': 12, "24h": 24, '36h': 36, "48h": 48, '60h': 60, "72h": 72, '84h': 84, "96h": 96, '108h': 108, "120h": 120, '132h': 132, '144h': 144, "0-96h": 0, "T0": 0, "T1": 24, "T2": 48, "T3": 72, "T4": 96}
    pcaAnalyser.pureGenotypeColName = "genotype"
    pcaAnalyser.addTimePointAsNumber(pcaAnalyser.table, timePointConversionFromStrToNumber, timePointColName=pcaAnalyser.timePointNameColName)
    geneColorMapperDict = pcaAnalyser.createTimePointDependentGeneMapper(pcaAnalyser.table, genotypeColorConversion)
    entriesColors = []
    for i, row in pcaAnalyser.table.iterrows():
        geneName = row["genotype"]
        timePointIdx = row[pcaAnalyser.numericTimePointColName]
        color = geneColorMapperDict[geneName].to_rgba(timePointIdx)
        entriesColors.append(color)
    pcaAnalyser.table[colorColumnName] = entriesColors

def analyseEngCotyledonsRegularityPooledPCA(pcXIdx: int = 0, pcYIdx: int = 1, justPoolGenotypes: bool = False, justPoolTimePoints: bool = False):
    tableFilename = "Results/combinedMeasures_Eng2021Cotyledons.csv"
    baseFilenameToSave = f"Results/regularityResults/PCA/PCA_Eng2021Cotyledons_pooled"
    labelNameConverterDict = {"lengthGiniCoeff": "Gini coefficient of length", "angleGiniCoeff": "Gini coefficient of angle",
                              "relativeCompleteness": "relative completeness", "lobyness": "lobyness"}
    columnsToAnalyse = ["angleGiniCoeff", "lengthGiniCoeff", "relativeCompleteness", "lobyness"]

    loadColorPalette = createLoadColorPalette()
    colorPalette = sns.color_palette("colorblind")
    genotypeColorConversion = {"col-0": colorPalette[7], "Oryzalin": colorPalette[8], "ktn1-2": colorPalette[0]}

    analyser = PCAAnalysis(tableFilename)
    addGenotypeTimePointDependentColor(analyser, genotypeColorConversion)
    analyser.AnalyseTable(columnsToAnalyse, numberOfComponents=4)
    title = None
    if justPoolGenotypes:
        allTimePoints = ["0h", "24h", "48h", "72h", "96h"]
        genotypes = []
        fig, ax = plotIndividualGenotypeTimePointPCAs(analyser, columnsToAnalyse, allTimePoints=allTimePoints, genotypes=genotypes, hideInnerAxisLabels=hideInnerAxisLabels)
        baseFilenameToSave += "_justGenotype"
    elif justPoolTimePoints:
        hideInnerAxisLabels = False
        allTimePoints = []
        genotypes = ["col-0", "Oryzalin", "ktn1-2"]
        genotypeNames = ["WT", "WT+Oryzalin", "$\it{ktn1}$-$\it{2}$"]
        fig, ax = plotIndividualGenotypeTimePointPCAs(analyser, columnsToAnalyse, allTimePoints=allTimePoints, genotypes=genotypes, hideInnerAxisLabels=hideInnerAxisLabels)
        if not hideInnerAxisLabels:
            xPos = 0.07
            genotypeYPositions = np.asarray([0.85, 0.55, 0.225])
        else:
            xPos = 0.07
            genotypeYPositions = [0.767, 0.5, 0.227]
        for g, yPos in zip(genotypeNames, genotypeYPositions):
            plt.gcf().text(xPos, yPos, g, fontsize="large", rotation="vertical", horizontalalignment="center", verticalalignment="center")
        baseFilenameToSave += "_justTime"
    else:
        scatterColor = analyser.table["color"]
        analyser.PlotBiPlot(pcXIdx=pcXIdx, pcYIdx=pcYIdx, loadColorPalette=loadColorPalette, title=title, scatterColor=scatterColor, saveOrShowKwargs={"showPlot": False})
        addLegend([labelNameConverterDict[label] for label in columnsToAnalyse], loadColorPalette)
    filenameToSave = baseFilenameToSave + f"{'' if pcXIdx == 0 and pcYIdx == 1 else f'_PC{pcXIdx + 1}VS{pcYIdx + 1}'}.png"
    saveOrShowKwargs = {"filenameToSave": filenameToSave, "showPlot": True, "dpi": 300}
    analyser.SaveOrShowFigure(**saveOrShowKwargs)

if __name__ == '__main__':
    for pcXIdx, pcYIdx in itertools.combinations(range(4), r=2):
        analyseEngCotyledonsRegularityIndividualPCA(pcXIdx=pcXIdx, pcYIdx=pcYIdx)
        analyseEngCotyledonsRegularityPooledPCA(pcXIdx=pcXIdx, pcYIdx=pcYIdx)
        analyseEngCotyledonsRegularityPooledPCA(pcXIdx=pcXIdx, pcYIdx=pcYIdx, justPoolTimePoints=True)
        analyseEngCotyledonsRegularityPooledPCA(pcXIdx=pcXIdx, pcYIdx=pcYIdx, justPoolGenotypes=True)
