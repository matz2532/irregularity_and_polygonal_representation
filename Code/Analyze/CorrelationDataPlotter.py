import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns
import pandas as pd

from matplotlib.offsetbox import AnchoredText

class CorrelationDataPlotter (object):

    def DoScatterPlotWithSeaborn(self, data, x, y, ax, color=None, alpha=0.3, xLim=None, yLim=None,
                      showPlot=False, title=None, textLocation="upper right",
                      savefig=False, filenameToSave="correlationPlot.png", fontSize=10,
                      showXLabel=True, showYLabel=True, xLabel="observed value", ylabel="predicted value",
                      showXTicks=True, showYTicks=True, doPearson=True, **kwargs):
        matplotlib.rcParams.update({"font.size":fontSize})
        sns.scatterplot(data=data, x=x, y=y, ax=ax, alpha=alpha, color=color, **kwargs)
        regressionStats = self.addRegressionLineAndTextFromData(ax, data, x, y, xLim=xLim, yLim=yLim, textLocation=textLocation, doPearson=doPearson)
        if not title is None:
            ax.set_title(title, {'fontsize':fontSize})
        if showXLabel:
            ax.set_xlabel(xLabel, size=fontSize)
        if showYLabel:
            ax.set_ylabel(ylabel, size=fontSize)
        ax.tick_params(left=showYTicks , bottom=showXTicks, axis="both", labelsize=fontSize)
        ax.set_xlim(xLim)
        ax.set_ylim(yLim)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if savefig:
            plt.savefig(filenameToSave, bbox_inches="tight")
            plt.close()
        else:
            if showPlot:
                plt.show()
        return regressionStats

    def doScatterPlot(self, x, y, ax, marker="o", color="g", xLim=None, yLim=None, alpha=0.3,
                      showPlot=False, labels="", textLocation="upper right",
                      savefig=False, filenameToSave="correlationPlot.png", fontSize=10,
                      showXLabel=True, showYLabel=True, xLabel="observed value", ylabel="predicted value",
                      showTitle=True, showXTicks=True, showYTicks=True, doPearson=True):
        matplotlib.rcParams.update({"font.size":fontSize})
        ax.scatter(x, y, marker=marker, color=color, alpha=alpha)
        self.addRegressionLineAndText(ax, x, y, xLim=xLim, yLim=yLim, textLocation=textLocation, doPearson=doPearson)
        if showTitle:
            ax.set_title(labels, {'fontsize':fontSize})
        if showXLabel:
            ax.set_xlabel(xLabel, size=fontSize)
        if showYLabel:
            ax.set_ylabel(ylabel, size=fontSize)
        ax.tick_params(left=showYTicks , bottom=showXTicks, axis="both", labelsize=fontSize)
        ax.set_xlim(xLim)
        ax.set_ylim(yLim)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if savefig:
            plt.savefig(filenameToSave, bbox_inches="tight")
            plt.close()
        else:
            if showPlot:
                plt.show()

    def addRegressionLineAndTextFromData(self, ax, data, x, y, xLim=None, yLim=None, textLocation="upper right", doPearson=True):
        x = data[x]
        y = data[y]
        return self.addRegressionLineAndText(ax, x, y, xLim=xLim, yLim=yLim, textLocation=textLocation, doPearson=doPearson)

    def addRegressionLineAndText(self, ax, x, y, xLim=None, yLim=None, showRegressionLine=True, lw=1, textLocation="upper right",
                                 doPearson=True, showRSquared=False, addLinearFormulaText=True, addRegressionText=True, 
                                 addFancyBox=True, manuallyAddText=False):
        if isinstance(x, pd.Series):
            x = x.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        isXNan = np.isnan(x)
        isYNan = np.isnan(y)
        if np.any(isXNan) or np.any(isYNan):
            indicesToRemove = []
            indicesToRemove.extend(list(np.where(isXNan)[0]))
            indicesToRemove.extend(list(np.where(isYNan)[0]))
            x = np.delete(x, indicesToRemove)
            y = np.delete(y, indicesToRemove)
        if np.any(np.isinf(x)):
            raise NotImplementedError(f"Not implemented how to handle infinite values in {x=}")
        if np.any(np.isinf((y))):
            raise NotImplementedError(f"Not implemented how to handle infinite values in {y=}")
        m, b = np.polyfit(x, y, 1)
        if xLim is None:
            xLim = [np.min(x), np.max(x)]
        if yLim is None:
            yLim = [np.min(y), np.max(y)]
        if doPearson:
            correlationAbbreviationName = "r"
            r, p = scipy.stats.pearsonr(x, y)
        else:
            correlationAbbreviationName = "rho"
            r, p = scipy.stats.spearmanr(x, y)
        if showRegressionLine:
            ax.plot(xLim, np.poly1d((m, b))(xLim), c="black", lw=lw, linestyle="dashed", zorder=10)
        textToAdd = ""
        if addLinearFormulaText or addRegressionText:
            if addLinearFormulaText:
                linearFormulaTxt = self.createLinearFormulaFunctionText(m, b)
                textToAdd += linearFormulaTxt
                if addRegressionText:
                    textToAdd += "\n"
            if addRegressionText:
                correlationTxt = self.createCorrelationText(r, p, correlationAbbreviationName=correlationAbbreviationName, showRSquared=showRSquared)
                textToAdd += correlationTxt
            if not manuallyAddText:
                if type(textLocation) == str:
                    anchoredText = AnchoredText(textToAdd, loc=textLocation, frameon=False)
                    ax.add_artist(anchoredText)
                else:
                    if textLocation is None:
                        xTextLocation = xLim[0] + 0.5 * (xLim[1] - xLim[0])
                        yTextLocationOffset = 0.2 * (yLim[1] - yLim[0])
                        yTextLocation = np.poly1d((m, b))(xTextLocation) + yTextLocationOffset
                        textLocation = (xTextLocation, yTextLocation)
                    if addFancyBox:
                        fancyBoxKwargs = dict(c=(0.9, 0.9, 0.9),
                                              bbox=dict(boxstyle="round", ec=(0, 0, 0, 0.5), fc=(0, 0, 0, 0.3)))
                    else:
                        fancyBoxKwargs = {}
                    ax.annotate(textToAdd, xy=textLocation, ha="center", va="center", fontsize="xx-small", **fancyBoxKwargs)
        fittingResults = {"slope":m, "intercept":b, "regression value":r, "p-value":p, "statistic":"Pearson correlation" if doPearson else "Spearman correlation"}
        if manuallyAddText:
            return fittingResults, textToAdd
        else:
            return fittingResults

    def createCorrelationText(self, r, p, correlationAbbreviationName="r", showRSquared=False):
        if showRSquared:
            correlationAbbreviationName = correlationAbbreviationName.capitalize() + "²"
            r = r**2
        correlationTxt = f"0:.{self.getRoundToValue(r)}f"
        correlationTxt = correlationAbbreviationName + " = {" + correlationTxt + "}"
        correlationTxt = correlationTxt.format(r)
        if p < 0.05:
            correlationTxt = r"$\bf{" + correlationTxt + r"}$"
        return correlationTxt

    def getRoundToValue(self, value, minDecimalPlaces=2):
        value = np.abs(value)
        if value == 0:
            return int(value)
        elif value >= 0.1:
            roundTo = minDecimalPlaces
        else:
            roundTo = np.abs(np.log10(value)-2)
        return int(roundTo)

    def createLinearFormulaFunctionText(self, m, b):
        mRoundingTxt = f"0:.{self.getRoundToValue(m)}f"
        biasRoundingTxt = f"1:.{self.getRoundToValue(b)}f"
        linearFormulaTxt = "f(x) = {" + mRoundingTxt + "}*x+{" + biasRoundingTxt + "}\n"
        linearFormulaTxt = linearFormulaTxt.format(m, b)
        return linearFormulaTxt

def correlateAreaVsIrregularitiesOfDataSet(dataSetname="Eng2021Cotyledons"):
    import itertools
    import sys

    sys.path.insert(0, "./Code/DataStructures/")
    sys.path.insert(0, "./Code/Analyze/")
    sys.path.insert(0, "./Images/")

    from InputData import GetResolutions
    from CorrelationDataPlotter import CorrelationDataPlotter

    if dataSetname == "Eng2021Cotyledons":
        genotypesResolutionDict = GetResolutions()
    else:
        genotypesResolutionDict = None
    filename = f"Results/combinedMeasures_{dataSetname}.csv"
    allGenotypeTableData = pd.read_csv(filename)
    colorPalette = sns.color_palette("colorblind")
    genotypeColorConversion = {"WT": colorPalette[7], "col-0": colorPalette[7], "WT_4dag": colorPalette[7], "Oryzalin": colorPalette[8], "WT+Oryzalin": colorPalette[8], "ktn": colorPalette[0], "ktn1-2": colorPalette[0],
                               "$\it{ktn1}$-$\it{2}$": colorPalette[0], "speechless": colorPalette[1]}
    genotypeColumnName = "genotype"
    timePointColumnName = "time point"
    potentialXValueColumnNames = ["angleGiniCoeff", "lengthGiniCoeff", "angleGiniCoeff_ignoringGuardCells", "lengthGiniCoeff_ignoringGuardCells"]
    potentialYValueColumnNames = ["labelledImageArea", "originalPolygonArea", "regularPolygonArea"]

    genotypeNames = pd.unique(allGenotypeTableData[genotypeColumnName])
    timePointNames = pd.unique(allGenotypeTableData[timePointColumnName])
    nrOfGenotypes, nrOfTimePoints = len(genotypeNames), len(timePointNames)

    titleConverter = {"Eng2021Cotyledons": "Eng 2021 cotyledons 0-96h", "Smit2023Cotyledons": "Smit 2023 cotyledons"}
    unabreviatedDatasetName = titleConverter[dataSetname] if dataSetname in titleConverter else dataSetname

    saveInsteadOfShowingFigure = True
    correlationPlotter = CorrelationDataPlotter()

    baseSize = np.array([4, 3]) * 1.5
    figsize = [baseSize[0] * nrOfTimePoints, baseSize[1] * nrOfGenotypes]
    for yValueColumnName, xValueColumnName in itertools.product(potentialYValueColumnNames, potentialXValueColumnNames):
        fig, ax = plt.subplots(nrOfGenotypes, nrOfTimePoints, figsize=figsize)
        ax = ax.ravel()
        i = 0
        for selectedGenotype, selectedTimePoint in itertools.product(genotypeNames, timePointNames):
            if genotypesResolutionDict is not None:
                assert selectedGenotype in genotypesResolutionDict, f"The {selectedGenotype=} is not in the genotypesResolutionDict keys {list(genotypesResolutionDict.keys())}"
                resolution = genotypesResolutionDict[selectedGenotype]
            else:
                resolution = 1
            isSelectedGenotype = allGenotypeTableData[genotypeColumnName] == selectedGenotype
            isSelectedTimePoint = allGenotypeTableData[timePointColumnName] == selectedTimePoint
            dataOfGenotype = allGenotypeTableData.loc[isSelectedGenotype & isSelectedTimePoint]
            assert xValueColumnName in dataOfGenotype.columns, f"The column {xValueColumnName} is not present in the table, only are present {dataOfGenotype.columns}"
            assert yValueColumnName in dataOfGenotype.columns, f"The column {yValueColumnName} is not present in the table, only are present {dataOfGenotype.columns}"
            xValues = dataOfGenotype[xValueColumnName]
            yValues = dataOfGenotype[yValueColumnName] * resolution
            sns.scatterplot(x=xValues, y=yValues, ax=ax[i], color=genotypeColorConversion[selectedGenotype])  # , label=selectedGenotype)
            correlationPlotter.addRegressionLineAndText(ax[i], x=xValues, y=yValues, showRSquared=True)
            if i % nrOfTimePoints != 0:
                ax[i].set_ylabel("")
            if np.ceil((i + 1) / nrOfTimePoints) != nrOfGenotypes:
                ax[i].set_xlabel("")
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            i += 1
        # plt.legend()
        if saveInsteadOfShowingFigure:
            plt.savefig(f"./Results/regularityResults/areaVs/{unabreviatedDatasetName} with {yValueColumnName} vs {xValueColumnName}.png", bbox_inches="tight", dpi=300)
            plt.close()
        else:
            plt.show()

def main():
    myCorrelationDataPlotter = CorrelationDataPlotter()
    txt = myCorrelationDataPlotter.createCorrelationText(0.05, 0.2)
    print(txt)
    txt = myCorrelationDataPlotter.createCorrelationText(0.005, 0.2)
    print(txt)
    txt = myCorrelationDataPlotter.createLinearFormulaFunctionText(0.5, 0.2)

if __name__ == '__main__':
    # correlateAreaVsIrregularitiesOfDataSet(dataSetname="Eng2021Cotyledons")
    correlateAreaVsIrregularitiesOfDataSet(dataSetname="Smit2023Cotyledons")
    # main()
