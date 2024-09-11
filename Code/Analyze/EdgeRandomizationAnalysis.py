import sys

import matplotlib.pyplot as plt

sys.path.insert(0, "./Code/DataStructures/")
from MultiFolderContent import MultiFolderContent

class EdgeRandomizationAnalysis:

    folderContents: MultiFolderContent = None
    currentSeed: int = 42
    junctionPositionsOfContent: {"contentName": 1}
    randomizationDifferencesPerContent: dict[str, list[list[float]]]
    # inner list of floats represents difference of original with randomization
    # outer list represents different entries from original

    def __init__(self, folderContentsFilename: str or MultiFolderContent):
        self.folderContents = MultiFolderContent(folderContentsFilename)

    def SetJunctionPositionsOfContent(self, junctionPositionsKey: str, folderContentsFilename: str or MultiFolderContent or None):
        if folderContentsFilename is not None:
            self.folderContents = MultiFolderContent(folderContentsFilename)
        assert self.folderContents is not None, f"The folder contents needs to be defined, when setting the junction positions of these contents."
        # <----- implement from here
        self.junctionPositionsOfContent = {}

    def RandomizeEdgesWithoutPlanarityCheck(self,
            randomizationSeed: int,
            junctionPositionsKey: str or None = None,
            compareToValuesKey: str or None = None
        ):
        if junctionPositionsKey is None:
            assert self.junctionPositionsOfContent is not None, f"You need to either specify the junction positions of the corresponding contents (name being key) or specify the junctionPositionsKey parameter."
        else:
            self.SetJunctionPositionsOfContent(junctionPositionsKey=junctionPositionsKey)
            self.randomizationDifferencesPerContent = {}
        self.currentSeed = randomizationSeed
        # <----- implement from here

    def AnalyzeRandomizationResults(self, saveProperties: dict or None = None, showPlot: bool = False):
        # <----- implement visualization here
        if saveProperties is not None:
            plt.savefig(**saveProperties)
        elif showPlot:
            plt.show()

def testFunctionality():
    dataSetName = "Eng2021Cotyledons" # "Smit2023Cotyledons" #
    junctionPositionsKey = "finalJunctionFilename"

    filename = f"Images/{dataSetName}/{dataSetName}.json"
    EdgeRandomizationAnalysis(filename)

if __name__ == '__main__':
    testFunctionality()