import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
# import pingouin as pg
import seaborn as sns
import sys

sys.path.insert(0, "./Code/DataStructures/")

from AnovaLetterPlotter import AnovaLetterPlotter
from MultiFolderContent import MultiFolderContent
from pathlib import Path

class BasePlotter (object):

    genotypeColorConversion=None
    labelNameConverterDict = {"lengthGiniCoeff":"Gini coefficient of length", "angleGiniCoeff":"Gini coefficient of angle"}

    def __init__(self, allFolderContentsFilename=None, resultsFolder=None,
                 allTimePoints=["0h", "24h", "48h", "72h", "96h"]):
        if not allFolderContentsFilename is None:
            self.multiFolderContent = MultiFolderContent(allFolderContentsFilename)
        else:
            self.multiFolderContent = None
        self.resultsFolder = resultsFolder
        self.allTimePoints = allTimePoints

    def SetGenotypeColorConversion(self, genotypeColorConversion):
        self.genotypeColorConversion = genotypeColorConversion

    def SetLabelNameConverterDict(self, labelNameConverterDict, extend=False):
        if extend:
            for k, v in labelNameConverterDict.items():
                self.labelNameConverterDict[k] = v
        else:
            self.labelNameConverterDict = labelNameConverterDict

    def SetRcParams(self, fontSize=None):
        if not fontSize is None:
            plt.rcParams["font.size"] = fontSize
        plt.rcParams["font.family"] = "Arial"
        mpl.rcParams['axes.spines.right'] = False
        mpl.rcParams['axes.spines.top'] = False

    def GetLabelOfMeasure(self, measureName, requirePresenceInConverter=True):
        assert not requirePresenceInConverter or measureName in self.labelNameConverterDict, f"The {measureName=} is not present as a key in the labelNameConverterDict choose one of {list(self.labelNameConverterDict.keys())} or use SetLabelNameConverterDict to define your own.\nAlternative set requirePresenceInConverter to False to return {measureName=} as label."
        if measureName in self.labelNameConverterDict:
            labelMeasureName = self.labelNameConverterDict[measureName]
        else:
            labelMeasureName = measureName
        return labelMeasureName

    def CreateBoxPlot(self, measureName, loadMeasureTableFromFilename=None,
                      extractMeassureFromFilenameUsingKey="regularityMeasuresFilename",
                      ignoreColumnValuesOf=None, orderByGenotype=False,
                      doShowPlot=True, removeLegend=False, testAgainstValue=None):
        if not loadMeasureTableFromFilename is None:
            measureResultsTidyDf = pd.read_csv(loadMeasureTableFromFilename)
            measureResultsTidyDf[measureName] = measureResultsTidyDf[measureName].astype(float)
        else:
            self.multiFolderContent.AddDataFromFilename(measureName, extractMeassureFromFilenameUsingKey)
            measureResultsTidyDf = self.multiFolderContent.GetTidyDataFrameOf([measureName], includeCellId=True)
            measureResultsTidyDf[measureName] = measureResultsTidyDf[measureName].astype(float)
        if not ignoreColumnValuesOf is None:
            for columnName, valueToIgnore in ignoreColumnValuesOf.items():
                measureResultsTidyDf = measureResultsTidyDf.iloc[np.isin(measureResultsTidyDf[columnName], valueToIgnore, invert=True), :]
        # visualise mean results
        plt.rcParams["font.size"] = 16
        fig, ax = plt.subplots(figsize=[1.2 * (1+len(self.allTimePoints)), 5.2], constrained_layout=True)
        timeToIntDict = dict(zip(self.allTimePoints, np.arange(len(self.allTimePoints))))
        measureResultsTidyDf["intTimePointColName"] = [timeToIntDict[t] for t in measureResultsTidyDf["time point"]]
        if orderByGenotype:
            g = sns.boxplot(x="genotype", y=measureName, data=measureResultsTidyDf, hue="time point", showmeans=True, meanprops={"markerfacecolor":"white", "markeredgecolor":"black"})
            for i, artist in enumerate(g.artists):
                artist.set_facecolor(self.genotypeColorConversion[i])
        else:
            g = sns.boxplot(x="time point", y=measureName, data=measureResultsTidyDf, hue="genotype", showmeans=True, meanprops={"markerfacecolor":"white", "markeredgecolor":"black"}, palette=self.genotypeColorConversion)
        g.legend(bbox_to_anchor=(0.8, 1.15), loc=2, borderaxespad=0)#(1.05, 1)
        if removeLegend:
            ax.get_legend().remove()
        if orderByGenotype:
            g.set_xlabel("genotype")
        else:
            g.set_xlabel("hours post dissection")
        labelMeasureName = self.GetLabelOfMeasure(measureName)
        g.set_ylabel(labelMeasureName)
        ylim = g.get_ylim()
        yOffset = 0.05 * (ylim[1]-ylim[0])
        anovaPlotter = AnovaLetterPlotter()
        if testAgainstValue is None:
            if orderByGenotype:
                anovaPlotter.AddAnovaMultiGroupLetters(ax, measureResultsTidyDf, "genotype", measureName, "time point", yOffset=yOffset)
            else:
                anovaPlotter.AddAnovaMultiGroupLetters(ax, measureResultsTidyDf, "time point", measureName, "genotype", yOffset=yOffset)
        else:
            if orderByGenotype:
                anovaPlotter.AddOneSampleTestOfYColumnAgainstValue(ax, measureResultsTidyDf, "genotype", measureName, "time point", testAgainstValue, yOffset=yOffset)
            else:
                anovaPlotter.AddOneSampleTestOfYColumnAgainstValue(ax, measureResultsTidyDf, "time point", measureName, "genotype", testAgainstValue, yOffset=yOffset)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if not self.resultsFolder is None:
            Path(self.resultsFolder).mkdir(parents=True, exist_ok=True)
            if orderByGenotype:
                plotTypeExtension = "_boxplot_orderingPerGenotpye.png"
            else:
                plotTypeExtension = "_boxplot.png"
            filenameToSave = self.resultsFolder + measureName + plotTypeExtension
            if testAgainstValue is None:
                between = []
                if len(measureResultsTidyDf["genotype"].unique()) > 1:
                    between.append("genotype")
                if len(measureResultsTidyDf["time point"].unique()) > 1:
                    between.append("time point")
                anovaResultsFilenameToSave = Path(filenameToSave).with_name(Path(filenameToSave).stem + "_anovaResults.csv")
                if len(between) > 0:
                    anovaResults = measureResultsTidyDf.anova(dv=measureName, between=between)
                    anovaResults.to_csv(anovaResultsFilenameToSave, index=False)
                anovaPlotter.SaveStatistic(anovaResultsFilenameToSave)
                anovaPlotter.SaveSampleNumbers(Path(anovaResultsFilenameToSave).parent, filenameExtension="sampleNumbers.csv")
            else:
                oneSampleTTestResultsFilename = filenameToSave.replace(".png", f"_oneSampleTTestVs_{testAgainstValue}.csv")
                anovaPlotter.SaveOneSampleTTestStatistic(exactFilenameToSave=oneSampleTTestResultsFilename)
            plt.savefig(filenameToSave, bbox_inches="tight")
            plt.close()
        else:
            if doShowPlot:
                plt.show()

    def CreateViolinPlot(self, measureName, loadMeasureTableFromFilename=None, extractMeassureFromFilenameUsingKey="regularityMeasuresFilename", doShowPlot=True, removeLegend=False):
        if not loadMeasureTableFromFilename is None:
            measureResultsTidyDf = pd.read_csv(loadMeasureTableFromFilename)
        else:
            self.multiFolderContent.AddDataFromFilename(measureName, extractMeassureFromFilenameUsingKey)
            measureResultsTidyDf = self.multiFolderContent.GetTidyDataFrameOf([measureName], includeCellId=True)
            measureResultsTidyDf[measureName] = measureResultsTidyDf[measureName].astype(float)
        # visualise mean results
        plt.rcParams["font.size"] = 16
        fig, ax = plt.subplots(figsize=[1.2 * (1+len(self.allTimePoints)), 5.2], constrained_layout=True)
        timeToIntDict = dict(zip(self.allTimePoints, np.arange(len(self.allTimePoints))))
        measureResultsTidyDf["intTimePointColName"] = [timeToIntDict[t] for t in measureResultsTidyDf["time point"]]
        # g = sns.boxplot(x="time point", y=measureName, data=measureResultsTidyDf, hue="genotype"))
        g = sns.violinplot(x="time point", y=measureName, data=measureResultsTidyDf, hue="genotype",
                          scatter_kws={"alpha":0.8, "s":60}, line_kws={"alpha":0.7, "lw":2})#"color":"black"
        g.legend(bbox_to_anchor=(0.8, 1.15), loc=2, borderaxespad=0)#(1.05, 1)
        if removeLegend:
            ax.get_legend().remove()
        g.set_xlabel("hours post dissection")
        labelMeasureName = self.GetLabelOfMeasure(measureName)
        g.set_ylabel(labelMeasureName)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if not self.resultsFolder is None:
            anovaResults = measureResultsTidyDf.anova(dv=measureName, between=["genotype", "time point"])
            Path(self.resultsFolder).mkdir(parents=True, exist_ok=True)
            anovaResultsFilenameToSave = self.resultsFolder + measureName + "_anovaResults.csv"
            anovaResults.to_csv(anovaResultsFilenameToSave, index=False)
            filenameToSave = self.resultsFolder + measureName + ".png"
            plt.savefig(filenameToSave, bbox_inches="tight")
            plt.close()
        else:
            if doShowPlot:
                plt.show()

    def CreatePairedPlot(self, measureName, loadMeasureTableFromFilename=None, extractMeassureFromFilenameUsingKey="regularityMeasuresFilename", doShowPlot=True, addTitle=False, setYLimMinZero=True):
        if not loadMeasureTableFromFilename is None:
            measureResultsTidyDf = pd.read_csv(loadMeasureTableFromFilename)
        else:
            self.multiFolderContent.AddDataFromFilename(measureName, extractMeassureFromFilenameUsingKey)
            measureResultsTidyDf = self.multiFolderContent.GetTidyDataFrameOf([measureName], includeCellId=True)
            measureResultsTidyDf[measureName] = measureResultsTidyDf[measureName].astype(float)
        # visualise mean results
        plt.rcParams["font.size"] = 16
        allGenotypes = ["col-0", "Oryzalin", "ktn1-2", "clasp-1"]
        nrOfGenotypes = len(allGenotypes)
        fig, ax = plt.subplots(1, 4, figsize=[nrOfGenotypes*1.2 * (1+len(self.allTimePoints)), 5.2])
        timeToIntDict = dict(zip(self.allTimePoints, np.arange(len(self.allTimePoints))))
        measureResultsTidyDf["intTimePointColName"] = [timeToIntDict[t] for t in measureResultsTidyDf["time point"]]
        measureResultsTidyDf["unique cell Id over time"] = measureResultsTidyDf["genotype"] + "_|_" + measureResultsTidyDf["replicate id"] + "_|_" + measureResultsTidyDf["cell id"].astype(str)
        yLims = []
        for i, genotype in enumerate(allGenotypes):
            currentAx = ax[i]
            df = measureResultsTidyDf.query(f"genotype == '{genotype}'")
            self.plotPairedPlot(df, measureName, ax=currentAx)
            yLim = currentAx.get_ylim()
            yLims.append(yLim)
            if addTitle:
                currentAx.set(title=genotype)
            if i > 0:
                self.hideYAxis(currentAx)
            else:
                labelMeasureName = self.GetLabelOfMeasure(measureName)
                currentAx.set(ylabel=labelMeasureName)
        self.setPairedPlotMajorYAxis(ax, yLims, setYLimMinZero=setYLimMinZero)
        plt.subplots_adjust(wspace=0)
        if not self.resultsFolder is None:
            anovaResults = measureResultsTidyDf.anova(dv=measureName, between=["genotype", "time point"])
            Path(self.resultsFolder).mkdir(parents=True, exist_ok=True)
            anovaResultsFilenameToSave = self.resultsFolder + measureName + "_anovaResults.csv"
            anovaResults.to_csv(anovaResultsFilenameToSave, index=False)
            filenameToSave = self.resultsFolder + measureName + "pairedPlot.png"
            plt.savefig(filenameToSave, bbox_inches="tight")
            plt.close()
        else:
            if doShowPlot:
                plt.show()

    def SaveOrShowFigure(self, filenameToSave=None, showPlot=True, dpi=300):
        if not filenameToSave is None:
            Path(filenameToSave).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filenameToSave, bbox_inches="tight", dpi=dpi)
            plt.close()
        else:
            if showPlot:
                plt.show()

    def hideYAxis(self, ax):
        # remove y-axis of all, but the first axes
        ax.spines['left'].set_visible(False)
        ax.set(yticklabels=[])
        ax.set(ylabel=None)
        ax.tick_params(left=False)

    def setPairedPlotMajorYAxis(self, ax, yLims, setYLimMinZero=True):
        yLims = np.asarray(yLims)
        unifiedYLim = [np.min(yLims[:, 0]), np.max(yLims[:, 1])]
        if setYLimMinZero:
            unifiedYLim[0] = 0
        for i in range(len(ax)):
            ax[i].set_ylim(unifiedYLim)
        ax[0].spines['left'].set_bounds(unifiedYLim[0], unifiedYLim[1])
        offset = self.my_floor(unifiedYLim[0], 2)
        ax[0].yaxis.set_major_locator(plt.MaxNLocator(8))#plt.IndexLocator(base=0.1, offset=offset))#plt.MultipleLocator(0.2))


    def my_floor(self, a, precision=0):
        return np.round(a - 0.5 * 10**(-precision), precision)

    def plotPairedPlotForAllMeasures(self, allMeasureNames, extractMeassureFromFilenameUsingKey="regularityMeasuresFilename"):
        self.multiFolderContent.AddDataFromFilename(allMeasureNames, extractMeassureFromFilenameUsingKey)
        genotypes = self.multiFolderContent.GetGenotypes()
        _, idxOfUniques = np.unique(genotypes, return_index=True)
        uniqueGenotypes = np.asarray(genotypes)[np.sort(idxOfUniques)]
        genotypeColorDict = {"col-0" : "black", "Oryzalin" : "yellow", "ktn1-2" : "blue", "clasp-1" : "red"}
        for measureName in allMeasureNames:
            self.CreateViolinPlot(measureName)

    def plotPairedPlot(self, df, measureName, ax=None):
        pg.plot_paired(data=df, dv=measureName, within='time point',
                       subject='unique cell Id over time', ax=ax, pointplot_kwargs={"alpha":0.4})
        plt.xlabel("hours post dissection")
        labelMeasureName = self.labelNameConverterDict[measureName]
        plt.ylabel(labelMeasureName)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

def main():
    dataBaseFolder = "Images/"
    resultsFolder = None # "Results/"
    allMeasureNames = ["lengthGiniCoeff"]
    folderContentsName = "allFolderContents.pkl"
    allFolderContentsFilename = dataBaseFolder + folderContentsName
    myBasePlotter = BasePlotter(allFolderContentsFilename)
    # myBasePlotter.CreateViolinPlot(allMeasureNames[3])
    myBasePlotter.CreatePairedPlot(allMeasureNames[3])

if __name__ == '__main__':
    main()
