import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import sys

### (1.) Answer the following questions: Of all n-gons with a given perimeter, the one with the largest area is regular (same angles and side lengths). How far are real world cells from regular?
sys.path.insert(0, "./Code/DataStructures/")
from MultiFolderContent import MultiFolderContent
from shapely import LinearRing
dataSetname = "Smit2023Cotyledons" # "Eng2021Cotyledons" #
filename = f"Images/{dataSetname}/{dataSetname}.json"
loadKwargs = {"convertDictKeysToInt": False}
mfc = MultiFolderContent(filename)

def calcToCircleWithPerimeterNormalizedArea(perimeter, area):
    areaOfCircle = perimeter ** 2 / (4 * np.pi)
    return area / areaOfCircle

def calcNormalizedToCirclePolygonalArea(c):
    selectedKey = "areaMeasuresPerCell"
    originalPolygonalAreaPerCell = c.LoadKeyUsingFilenameDict(selectedKey, **loadKwargs)["originalPolygonArea"]
    originalPolygonalArea = list(originalPolygonalAreaPerCell.values())
    selectedKey = "orderedJunctionsPerCellFilename"
    orderedJunctionsPerCell = c.LoadKeyUsingFilenameDict(selectedKey)
    resolution = c.GetResolution()
    if resolution is None:
        resolution = 1
    perimeterOfPolygonizedCells = [LinearRing(junctionPositions).length * resolution for junctionPositions in orderedJunctionsPerCell.values()]
    normalizedArea = [calcToCircleWithPerimeterNormalizedArea(p, a) for p, a in zip(perimeterOfPolygonizedCells, originalPolygonalArea)]
    return normalizedArea

def calcNumberOfNeighborsAndMeanPlusStdNormAreasFor(mfc):
    pooledNormalizedPolygonalAreas, pooledNeighborsPerCell = [], []
    for c in mfc:
        normalizedAreaPolygonalArea = calcNormalizedToCirclePolygonalArea(c)
        pooledNormalizedPolygonalAreas.extend(normalizedAreaPolygonalArea)
        selectedKey = "orderedJunctionsPerCellFilename"
        orderedJunctionsPerCell = c.LoadKeyUsingFilenameDict(selectedKey)
        numberOfNeighborsPerCell = [len(junctionPositions) for junctionPositions in orderedJunctionsPerCell.values()]
        pooledNeighborsPerCell.extend(numberOfNeighborsPerCell)
    existingNumberOfNeighbors = np.unique(pooledNeighborsPerCell)
    stdNormalizedPolygonalAreaWithNeighbors, meanNormalizedPolygonalAreaWithNeighbors = [], []
    for n in existingNumberOfNeighbors:
        isSelectedCell = pooledNeighborsPerCell == n
        selectedArea = np.array(pooledNormalizedPolygonalAreas)[isSelectedCell]
        meanNormalizedPolygonalAreaWithNeighbors.append(np.mean(selectedArea))
        stdNormalizedPolygonalAreaWithNeighbors.append(np.std(selectedArea))
    return existingNumberOfNeighbors, meanNormalizedPolygonalAreaWithNeighbors, stdNormalizedPolygonalAreaWithNeighbors
allGenotypeData = {}
for genotype in mfc.GetGenotypes():
    genotypeTissues = mfc.GetFolderContentsOfGenotype(genotype)
    dataOfGenotype = calcNumberOfNeighborsAndMeanPlusStdNormAreasFor(genotypeTissues)
    allGenotypeData[genotype] = dataOfGenotype

allExistingNumberOfNeighbors = np.concatenate([data[0] for data in allGenotypeData.values()])
listOfNs = np.arange(3, np.max(allExistingNumberOfNeighbors)+1)
perimeter = 1
areaOfCircle = lambda p: p**2 / (4*np.pi)
areaOfSquare = lambda p: p**2 / 16
areaOfRegularNGon = lambda p, n: p**2 / (4*n*np.tan(np.deg2rad(180/n)))
calculatedAreaOfCircle = areaOfCircle(perimeter)

colorPalette = sns.color_palette("colorblind")
genotypeColorConversion = {"WT": colorPalette[7], "col-0": colorPalette[7], "WT_4dag": colorPalette[7], "Oryzalin": colorPalette[8], "WT+Oryzalin": colorPalette[8], "ktn": colorPalette[0], "ktn1-2": colorPalette[0], "$\it{ktn1}$-$\it{2}$": colorPalette[0], "speechless": colorPalette[1]}

plt.scatter(listOfNs, [areaOfRegularNGon(perimeter, n)/calculatedAreaOfCircle for n in listOfNs], label="area of regular n-gon", c=colorPalette[2])
for genotype, dataOfGenotype in allGenotypeData.items():
    existingNumberOfNeighbors, meanNormalizedPolygonalAreaWithNeighbors, stdNormalizedPolygonalAreaWithNeighbors = dataOfGenotype
    plt.errorbar(existingNumberOfNeighbors, meanNormalizedPolygonalAreaWithNeighbors, yerr=stdNormalizedPolygonalAreaWithNeighbors,
                 label=f"{genotype} 0-96h", linestyle='none', marker="o", c=genotypeColorConversion[genotype] if genotype in genotypeColorConversion else "black")
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
titleConverter = {"Eng2021Cotyledons": "Eng 2021 cotyledons 0-96h", "Smit2023Cotyledons": "Smit 2023 cotyledons"}
unabreviatedDatasetName = titleConverter[dataSetname] if dataSetname in titleConverter else dataSetname
plt.title(f"Comparison of regular n-gons and {unabreviatedDatasetName} cotyledons")
plt.ylabel("Normalized area\ncompared to circle of same perimeter")
plt.xlabel("# neighbors / edges")
plt.legend()
# plt.show()
plt.savefig(f"./Results/ratioResults/Normalized area vs number of edges of regular n-gons and {unabreviatedDatasetName}.png", bbox_inches="tight", dpi=300)
plt.close()