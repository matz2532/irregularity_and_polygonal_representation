import numpy as np
import skimage
import skimage.io
import sys

class PreProcessorDummy (object):

    validLabels=None
    contoursOfValdidLabels=None
    contoursOfValdidLabels=None

    def __init__(self, labelledImageFilename=None, labels=None, contourFilename=None,
                 contourOffsetCorrection=np.asarray([-1, -1])):
        # contourFilename is only used specify the labels to use from the labelled image in VisGraph
        self.originalImage = None
        self.contourOffsetCorrection = contourOffsetCorrection
        if labelledImageFilename is None:
            self.labelledImage = None
        else:
            if labelledImageFilename[-4:] == ".npy":
                self.labelledImage = np.load(labelledImageFilename)
            else:
                self.labelledImage = skimage.io.imread(labelledImageFilename)
        if not labels is None:
            self.labels = labels
        else:
            self.labels = np.unique(self.labelledImage)
            self.labels = self.labels[np.isin(self.labels, 0, invert=True)]
        if not contourFilename is None:
            self.validLabels, self.contoursOfValdidLabels = self.interpretContourFile(contourFilename)
            self.labels = self.validLabels
            self.contoursOfValdidLabels = self.validateContourForDoubles(self.contoursOfValdidLabels)
            if not self.contourOffsetCorrection is None:
                self.contoursOfValdidLabels = self.correctContours(self.contoursOfValdidLabels, self.contourOffsetCorrection)
        self.labeledImage = self.labelledImage
        self.skeletonImage = None
        self.branchlessSkeleton = None

    def interpretContourFile(self, contourFilename, headerOffSet=4):
        #if file contains empty places for example '' breaks
        file = open(contourFilename, "r")
        validLabels = []
        contours = {}
        lineNr = 0 - headerOffSet
        xCoordinates = None
        yCoordinates = None
        for line in file:
            if lineNr >= 0:
                splitLine = line.split("\t")
                if lineNr % 3 == 0:
                    xCoordinates = splitLine[1:]
                    xCoordinates = self.deleteUnvalidEntriesFrom(xCoordinates, unvalidEntries=["","\n"])
                elif lineNr % 3 == 1:
                    try:
                        label = int(splitLine[0])
                    except Error as e:
                        print("The Contour is probably not starting in the {} line".format(headerOffSet))
                        print(e)
                        sys.exit(1)
                    yCoordinates = splitLine[1:]
                    validLabels.append(label)
                    yCoordinates = self.deleteUnvalidEntriesFrom(yCoordinates, unvalidEntries=["","\n"])
                    assert len(yCoordinates) == len(xCoordinates), "The number of x- and y-cooridnates is not equal."
                    transposedContour = np.asarray([yCoordinates, xCoordinates], dtype=int).reshape((2,len(xCoordinates)))
                    contours[label] = transposedContour.T
            lineNr += 1
        for label, contour in contours.items():
            if list(contour[0, :]) == list(contour[-1, :]):
                contours[label] = contour[:-1, :]
        return validLabels, contours

    def validateContourForDoubles(self, contours):
        for label, contour in contours.items():
            closeDuplicates = self.findCloseDuplicatePixels(contour)
            if len(closeDuplicates) > 0:
                contour = self.correctContour(contour, closeDuplicates)
                contours[label] = contour
        return contours

    def findCloseDuplicatePixels(self, contour, distance=200):
        closeDuplicates = []
        nrOfCoordinates = contour.shape[0]
        for i in range(nrOfCoordinates-1):
            for j in range(i+1, nrOfCoordinates): #in range(i-distance, i):#
                cordI = contour[i,:]
                cordJ = contour[j,:]
                if list(cordI) == list(cordJ):
                    if j < 0:
                        j = nrOfCoordinates - j
                    closeDuplicates.append([i, j]) #closeDuplicates.append([j, i]) #
        return closeDuplicates

    def correctContour(self, contour, closeDuplicates, run=0):
        if len(closeDuplicates) > 1:
            startIdx = np.argmax([np.abs(i-j) for i, j in closeDuplicates])
        else:
            startIdx = 0
        start, end = closeDuplicates[startIdx]
        if start < 20 and end > contour.shape[0]-20:
            correctedContour = contour[start:end, :]
        else:
            beforeStartCoord = contour[:start, :]
            afterEndCoord = contour[end:, :]
            correctedContour = np.concatenate([beforeStartCoord, afterEndCoord], axis=0)
        closeDuplicates = self.findCloseDuplicatePixels(correctedContour)
        if len(closeDuplicates) > 0:
            run += 1
            assert run <= 10, "correctContour run {} times but there are still duplicates. Something went wrong.".format(run)
            correctedContour = self.correctContour(correctedContour, closeDuplicates, run)
        return correctedContour

    def deleteUnvalidEntriesFrom(self, vector, unvalidEntries=["","\n"]):
        vector = np.asarray(vector)
        for unvalidEntry in unvalidEntries:
            vector = np.delete(vector, np.where(vector == unvalidEntry)[0])
        vector = list(vector)
        return vector

    def correctContours(self, contoursOfCells, offset):
        offset = np.asarray(offset)
        for cellId, contour in contoursOfCells.items():
            contoursOfCells[cellId] += offset
        return contoursOfCells

    def GetOriginalImageFilename(self):
        return self.originalImageFilename

    def GetOriginalImage(self):
        return self.originalImage

    def GetSkeletonImage(self):
        # only needed for three-way junction detection
        return None

    def GetBranchlessSkeleton(self):
        return None

    def GetLabelledImage (self):
        return self.labelledImage

    def GetLabels(self):
        return self.labels

    def GetValidPavementCellLabels(self):
        return self.validLabels

    def GetContoursOfValdidLabels(self):
        return self.contoursOfValdidLabels

def main():
    import pickle
    sys.path.insert(0, "../../GraVisGUI/SourceCode/")
    sys.path.insert(0, "./Code/")
    sys.path.insert(0, "../Images/")
    from InputData import GetInputData
    from ShapeGUI import VisGraph
    genotype = "col-0"
    replicateId = "20170501 WT S2"
    tissueTimePointIdx = 0
    tmpFolder = "./Results/Test/VisGraphResults/"
    tissueTimePoint = ["0h"]
    inputData = GetInputData()
    folder = "../Images/{}/{}/{}/".format(genotype, replicateId, tissueTimePoint[tissueTimePointIdx])
    contourFilename = folder + inputData[genotype][replicateId]["contourFilename"][tissueTimePointIdx]
    labelledImageFilename  = folder + inputData[genotype][replicateId]["labelledImageFilename"][tissueTimePointIdx]
    originalImageFilename = None
    mode = "rb"
    with open(tmpFolder + "PreProcessorDummy.pkl", mode) as fh:
        if "r" in mode:
            myPreProcessorDummy = pickle.load(fh)
        else:
            myPreProcessorDummy = PreProcessorDummy(labelledImageFilename, contourFilename=contourFilename)
            pickle.dump(myPreProcessorDummy, fh)
    print(myPreProcessorDummy)
    import matplotlib.pyplot as plt
    orderedContours = myPreProcessorDummy.GetContoursOfValdidLabels()
    labelledImage = myPreProcessorDummy.GetLabelledImage()
    print("orderedContours", list(orderedContours.keys()))
    print("labelledImage", np.unique(labelledImage))
    labels = myPreProcessorDummy.GetLabels()
    print("labels", labels)
    filename = None
    preprocessedImage = myPreProcessorDummy
    resolution = 0.221
    outputFolder = tmpFolder
    plotLobeOutput = True
    roiFileList = False
    fileList = None
    plotIntermediate = True
    roiInput = False
    with open(tmpFolder + "myVisGraph.pkl", mode) as fh:
        if "r" in mode:
            myVisGraph = pickle.load(fh)
        else:
            myVisGraph = VisGraph(filename, preprocessedImage, plotIntermediate, resolution, outputFolder, plotLobeOutput, roiInput, roiFileList, fileList, saveTable=False)
            pickle.dump(myVisGraph, fh)
    visGraphs = myVisGraph.visibilityGraphs
    print("visGraphs keys:", list(visGraphs.keys()))
    dataBaseFolder = "Data/Eng 2021 time lines/"
    imageBaseFolder = "../Images/"
    restultsBaseFolder = "Results/Eng 2021 time lines/"
    allFolderContentsFilename = "Results/Test/allFolderContents_WT_S2_tmp.pkl"
    from MultiFolderContent import MultiFolderContent
    mfc = MultiFolderContent(allFolderContentsFilename)
    fc0h = list(mfc)[0]
    print(fc0h.GetTissueName())
    print(fc0h)

if __name__ == '__main__':
    main()
