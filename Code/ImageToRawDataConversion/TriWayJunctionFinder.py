import itertools
import skimage.morphology as mp
import numpy as np

class TriWayJunctionFinder (object):

    def __init__(self, labelledImage=None, cellsToIgnore=[0]):
        self.labelledImage = labelledImage
        self.cellsToIgnore = cellsToIgnore
        if not self.labelledImage is None:
            self.triWayJunctionPositions = self.calcTriWayJunctionPositions(self.labelledImage, self.cellsToIgnore)
            self.correspondingCells = self.findCorrespondingCells(self.triWayJunctionPositions, self.labelledImage, self.cellsToIgnore)

    def GetCorrespondingCells(self):
        return self.correspondingCells

    def GetJunctionPositionToCellDict(self):
        return {tuple(self.triWayJunctionPositions[i]): self.correspondingCells[i] for i in range(len(self.correspondingCells))}

    def GetTriWayJunctionPositions(self):
        return self.triWayJunctionPositions

    def calcTriWayJunctionPositions(self, labelledImage, cellsToIgnore=[0]):
        summedContourImg = np.zeros_like(labelledImage)
        validCells = np.unique(labelledImage)
        validCells = validCells[np.isin(validCells, cellsToIgnore, invert=True)]
        for cellId in validCells:
            contourImg = self.calcContourImageForFromLabelledImage(labelledImage, cellId)
            summedContourImg += contourImg
        boundaryPointsImg = summedContourImg.copy()
        boundaryPointsImg[boundaryPointsImg > 1] = 0
        boundaryPointsImg = mp.dilation(boundaryPointsImg, mp.disk(1))
        summedContourImg += boundaryPointsImg
        junctionPositions = np.asarray(np.where(summedContourImg==3)).T
        return junctionPositions

    def calcContourImageForFromLabelledImage(self, labelledImage, cellId):
        img = np.zeros_like(labelledImage)
        img[np.where(labelledImage == cellId)] = 1
        dilatedImg = mp.dilation(img, np.ones((3, 3)))
        contourImg = dilatedImg - img
        return contourImg

    def findCorrespondingCells(self, triWayJunctionPositions, labelledImage, cellsToIgnore=[0]):
        shape = labelledImage.shape
        correspondingCells = []
        for x,y in triWayJunctionPositions:
            xRange = self.clipRange(x, max=shape[0])
            yRange = self.clipRange(y, max=shape[1])
            windowSelection = np.asarray(list(itertools.product(xRange, yRange)))
            cellIds = np.unique(labelledImage[windowSelection[:, 0], windowSelection[:, 1]])
            validCellIds = cellIds[np.isin(cellIds, cellsToIgnore, invert=True)]
            correspondingCells.append(validCellIds)
        return correspondingCells

    def clipRange(self, x, max, width=1, min=0):
        givenRange = np.arange(x-width, x+width+1)
        isBellow = givenRange < max
        isEqualOrAbove = givenRange >= min
        return givenRange[isBellow & isEqualOrAbove]

def main():
    import matplotlib.pyplot as plt
    import skimage.io
    folder = "./Images/col-0/20170327 WT S1/0h/"
    labelledImageName = "F_BOREALIS_MTs_LTi6b-GFP mChTUA5 WT 0h S1_SMP-based (3) MIP_WATERSHED_CLEAN.TIF"
    originalImageName = "PM_LTi6b-GFP mChTUA5 WT 0h S1_SMP-based (3) MIP.TIF"
    originalImage = skimage.io.imread(folder + originalImageName)
    labelledImage = skimage.io.imread(folder + labelledImageName)
    myTriWayJunctionFinder = TriWayJunctionFinder(labelledImage)
    print(myTriWayJunctionFinder.GetJunctionPositionToCellDict())
    plt.imshow(labelledImage)
    plt.show()

if __name__ == '__main__':
    main()
