import json
import numpy as np
import skimage.morphology
import sys
import warnings

sys.path.insert(0, "./Code/DataStructures/")

from FolderContent import FolderContent

class LabelledImageToGraphConverter (object):

    # default parameters to ensure it was properly loaded
    labelledImage=None
    # default keys when providing a folder content of tissue
    labelledImageFolderContentExtractedContentKey="labelledImage"
    labelledImageFolderContentFilenameKey="labelledImageFilename"
    # labelled image related parameters
    # in case the labelled image value 1 is reserved for the middle lamella -> valueToConvertCellIdToProperLabelInImage=+1 and cellLabelsToIgnore=[0, 1]
    valueToConvertCellIdToProperLabelInImage=+0 # to convert cell id to pixel values in labelled image
    #                                             (in case the value 1 is reserved for the cell contour represented as a skeleton,
    #                                              we need to increment the cell id, which starts from 1 by +1 to find the appropriate cell label of 2)
    cellLabelsToIgnore=[0] # background label 0 should be ignored

    def __init__(self, labelledImageOrFilename=None, folderContent=None, selectedCellIds=[], runOnInit=True):
        self.SetLabelledImageProperties(labelledImageOrFilename, folderContent, selectedCellIds)
        if runOnInit:
            if not self.labelledImageOrFilename is None or not self.folderContent is None:
                self.ExtractAdjacencyListFromLabelledImage()

    def SetLabelledImageProperties(self, labelledImageOrFilename, folderContent, selectedCellIds=None):
        self.labelledImageOrFilename = labelledImageOrFilename
        self.folderContent = folderContent
        if not selectedCellIds is None:
            self.selectedCellIds = selectedCellIds

    def GetAdjacencyList(self):
        return self.adjacencyList

    def GetDefaultSelectedCellLabels(self):
        assert not self.labelledImage is None, f"You can not set default selected cell labels, when the labelled image is not properly defined.\n{self.labelledImageOrFilename=} {self.folderContent=}"
        selectedCellLabels = np.unique(self.labelledImage)
        selectedCellLabels = selectedCellLabels[np.isin(selectedCellLabels, self.cellLabelsToIgnore, invert=True)]
        return selectedCellLabels

    def ExtractAdjacencyListFromLabelledImage(self, labelledImageOrFilename=None, folderContent=None, selectedCellIds=None):
        if not labelledImageOrFilename is None or not folderContent is None:
            self.SetLabelledImageProperties(labelledImageOrFilename, folderContent, selectedCellIds)
        self.labelledImage = self.loadLabelledImage()
        self.adjacencyList = self.extractAdjacency()
        return self.adjacencyList

    def SaveAdjacencyList(self, filenameToSaveAs=None, dataBaseFolder="",
                          baseAdjacencyName="labelledImageAdjacencyList.json",
                          adjacencyFilenameKey="labelledImageAdjacencyList"):
        if filenameToSaveAs is None:
            if self.folderContent is None:
                filenameToSaveAs = dataBaseFolder + baseAdjacencyName
            else:
                folderExtension = self.folderContent.GetFolder()
                filenameToSaveAs = dataBaseFolder + folderExtension + baseAdjacencyName
        if not self.adjacencyList is None:
            with open(filenameToSaveAs, "w") as fh:
                json.dump(self.adjacencyList, fh)
            if not self.folderContent is None:
                self.folderContent.AddDataToFilenameDict(filenameToSaveAs, adjacencyFilenameKey)

    def loadLabelledImage(self):
        assert not self.labelledImageOrFilename is None or not self.folderContent is None, f"To load the labelled image to create the tissues graph representation you need to either give the labelled image or its filename ({labelledImageOrFilename=}), or a folder content of the tissue ({folderContent=})."
        if self.labelledImageOrFilename is None:
            if self.folderContent.IsKeyInExtractedFilesDict(self.labelledImageFolderContentExtractedContentKey):
                labelledImage = self.folderContent.GetExtractedFilesKeyValues(self.labelledImageFolderContentExtractedContentKey)
            elif self.folderContent.IsKeyInFilenameDict(self.labelledImageFolderContentFilenameKey):
                labelledImage = self.folderContent.LoadKeyUsingFilenameDict(self.labelledImageFolderContentFilenameKey)
            else:
                print(f"The folder content did neither contain the labelled image as a content (using key={self.labelledImageFolderContentExtractedContentKey}) nor is the labelled image filename present (using key={self.labelledImageFolderContentFilenameKey}) using the tissue:\n{self.folderContent}")
                sys.exit(1)
        elif type(self.labelledImageOrFilename) == str:
            labelledImage = FolderContent().loadFile(self.labelledImageOrFilename)
        else:
            labelledImage = self.labelledImageOrFilename
        return labelledImage

    def extractAdjacency(self, expandDilationRadiusBy=2):
        if self.selectedCellIds is None:
            selectedCellLabels = self.GetDefaultSelectedCellLabels()
        else:
            if len(self.selectedCellIds) == 0:
                selectedCellLabels = self.GetDefaultSelectedCellLabels()
            else:
                selectedCellLabels = np.array(self.selectedCellIds) + self.valueToConvertCellIdToProperLabelInImage
        isSelectedLabelPresent = np.isin(selectedCellLabels, self.labelledImage)
        if not np.all(isSelectedLabelPresent):
            warnings.warn(f"The selectedCellLabels {np.array(selectedCellLabels)[np.invert(isSelectedLabelPresent)]} are not present in the labelled image, which only contains the pixel values {np.unique(self.labelledImage)}.\nThis will cause the not present labels to have no neighbors.")
        dilationExpansion = 1 + 2 * expandDilationRadiusBy
        dilationMask = np.ones((dilationExpansion, dilationExpansion))
        adjacencyList = {}
        contourDict = self.folderContent.LoadKeyUsingFilenameDict("cellContours", convertDictKeysToInt=True, convertDictValuesToNpArray=True)
        for cellLabel in selectedCellLabels:
            highlightedCellInImg = np.zeros_like(self.labelledImage)
            highlightedCellInImg[self.labelledImage == cellLabel] = 1
            dilatedCellImg = skimage.morphology.dilation(highlightedCellInImg, dilationMask)
            dilatedCellImg -= highlightedCellInImg
            potentialNeighborPositions = np.where(dilatedCellImg == 1)
            neighboringLabels = np.unique(self.labelledImage[potentialNeighborPositions])
            neighboringLabels = neighboringLabels[np.isin(neighboringLabels, self.cellLabelsToIgnore, invert=True)]
            cellId = cellLabel - self.valueToConvertCellIdToProperLabelInImage
            neighboringIds = neighboringLabels - self.valueToConvertCellIdToProperLabelInImage
            adjacencyList[int(cellId)] = [int(i) for i in neighboringIds]
        return adjacencyList

def main():
    import matplotlib.pyplot as plt
    import networkx as nx
    from MultiFolderContent import MultiFolderContent
    dataBaseFolder = "Images/"
    folderContentsName = "Eng2021Cotyledons.pkl"
    allFolderContentsFilename = dataBaseFolder + folderContentsName
    multiFolderContent = MultiFolderContent(allFolderContentsFilename)
    contourDict = list(multiFolderContent)[0].LoadKeyUsingFilenameDict("cellContours", convertDictKeysToInt=True, convertDictValuesToNpArray=True)
    selectedCellLabels = np.sort(list(contourDict.keys()))
    myLabelledImageToGraphConverter = LabelledImageToGraphConverter(folderContent=list(multiFolderContent)[0], selectedCellIds=selectedCellLabels)
    adjacencyList = myLabelledImageToGraphConverter.GetAdjacencyList()
    myLabelledImageToGraphConverter.SaveAdjacencyList(dataBaseFolder=dataBaseFolder)
    # multiFolderContent.UpdateFolderContents()
    print(adjacencyList)
    graph = nx.from_dict_of_lists(adjacencyList)
    nx.draw(graph)
    plt.show()

if __name__ == '__main__':
    main()
