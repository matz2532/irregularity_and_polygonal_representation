import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage.morphology

class JunctionCreator:

    verbose=0

    def __init__(self, labelledImage, filenameToSaveJunctions=None, graVisTypeLabelledImg=False, backgroundPixel=None):
        self.labelledImage = labelledImage
        self.filenameToSaveJunctions = filenameToSaveJunctions
        self.graVisTypeLabelledImg = graVisTypeLabelledImg
        self.triWayJunctions, self.skeletonImage = self.createSkeletonImgAndJunctionPositon(backgroundPixel=backgroundPixel)
        if not self.filenameToSaveJunctions is None:
            self.SaveTriWayJunctions(self.filenameToSaveJunctions)

    def createSkeletonImgAndJunctionPositon(self, backgroundPixel=None):
        if backgroundPixel is None:
            if self.graVisTypeLabelledImg:
                # backgroundPixel is 0, and 1 for GraVis based labelled images
                backgroundPixel = [0, 1]
            else:
                backgroundPixel = [0]
        labelRange = np.unique(self.labelledImage)
        labelRange = labelRange[np.isin(labelRange, backgroundPixel, invert=True)]
        summedDilationImg = np.zeros_like(self.labelledImage)
        for label in labelRange:
            dialtedContourImg = self.calcDilatedContourImage(label)
            summedDilationImg += dialtedContourImg
        skeletonImage = summedDilationImg.copy()
        skeletonImage[skeletonImage>1] = 1
        dialtedBgImg = self.calcDilatedBackgroundImgExludingSkeletonImage(backgroundPixel, skeletonImage)
        summedDilationImg += dialtedBgImg
        x, y = np.where(summedDilationImg>2)
        triWayJunctions = np.asarray([x,y]).T
        return triWayJunctions, skeletonImage

    def calcDilatedContourImage(self, label):
        contourImg = self.calcContourImage(label)
        dialtedContourImg = skimage.morphology.dilation(contourImg, np.ones((3,3)))
        dialtedContourImg -= contourImg
        return dialtedContourImg

    def calcContourImage(self, label):
        xOfLabel, yOfLabel = np.where(self.labelledImage==label)
        contourImg = np.zeros_like(self.labelledImage)
        for x, y in zip(xOfLabel, yOfLabel):
            contourImg[x, y] = 1
        return contourImg

    def calcDilatedBackgroundImgExludingSkeletonImage(self, backgroundPixel, skeletonImage):
        bgImage = np.zeros_like(self.labelledImage)
        for bg in backgroundPixel:
            contourImg = self.calcContourImage(bg)
            bgImage += contourImg
        bgImage[bgImage>1] = 1
        bgImage -= skeletonImage
        dialtedBgImg = skimage.morphology.dilation(bgImage, np.ones((3,3)))
        dialtedBgImg -= bgImage
        dialtedBgImg[dialtedBgImg>1] = 1
        labeledImage, labels = scipy.ndimage.label(dialtedBgImg)
        labels, counts = np.unique(labeledImage, return_counts=True)
        if len(labels) > 2:
            labelsToBeRemoved = labels[2:]
            if self.verbose >= 1:
                print("there are mutliple segment boarders and {} would be removed. Please confirm that they are inside the tissue and can therefore be removed.".format(labelsToBeRemoved))
                if self.verbose >= 2:
                    plt.imshow(labeledImage)
                    plt.show()
            for label in labelsToBeRemoved:
                x, y = np.where(labeledImage == label)
                dialtedBgImg[x, y] = 0
                labeledImage[x, y] = 0
            if self.verbose >= 1:
                print("resulting in this tissue outline.")
                plt.imshow(labeledImage)
                plt.show()
        return dialtedBgImg

    def SaveTriWayJunctions(self, filenameToSaveJunctions):
        np.save(filenameToSaveJunctions, self.triWayJunctions)

    def VisualiseJunctionsWithImage(self, img, junctions=None, fillJunctionsWith=255):
        if junctions is None:
            junctions = self.triWayJunctions
        labelledImgWithJunctions = img.copy()
        for x, y in junctions:
            labelledImgWithJunctions[x, y] = fillJunctionsWith
        plt.imshow(labelledImgWithJunctions)
        plt.show()

    def GetSkeletonImage(self):
        return self.skeletonImage

    def GetTriWayJunctions(self):
        return self.triWayJunctions

def main():
    import skimage.io
    folder = "./Images/col-0/20170501 WT S2/0h/"
    folderToSave = folder
    labelledImageName = "F_BOREALIS_MTs_LTi6b-GFP mChTUA5 0h S2_SMP_WATERSHED_CLEAN.TIF"
    labelledImage = skimage.io.imread(folder + labelledImageName)
    filenameToSaveJunctions = folderToSave + "baseJunctions.npy"
    myCorrector = JunctionCreator(labelledImage, filenameToSaveJunctions=filenameToSaveJunctions)
    junctions = np.load(filenameToSaveJunctions)
    print(junctions)
    myCorrector.VisualiseJunctionsWithImage(labelledImage, junctions)

if __name__ == '__main__':
    main()
