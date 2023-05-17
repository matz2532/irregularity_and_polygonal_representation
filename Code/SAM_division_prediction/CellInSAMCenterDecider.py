import numpy as np
import pandas as pd
import sys
import warnings

from pathlib import Path

class CellInSAMCenterDecider (object):

    def __init__(self, geometryData, centerCellLabels, centerRadius, skipFooter=4):
        # ----- ATTENTION last 4 lines need to be skipped, -----
        # ----- make sure this does not change with MorphoGraphX Version before applying script -----
        self.geometryDataFilename = geometryData
        self.geometryData = self.loadGeometricData(geometryData, skipFooter=skipFooter)
        self.centerCellLabels = np.asarray(centerCellLabels)
        self.centerRadius = centerRadius
        self.centerPosition = self.calculateCenterPosition(self.centerCellLabels)
        self.distanceOfCellsToCenter = self.calculateDistanceToCenter(self.centerPosition)
        self.centralCells, self.nonCentralCellLabels = self.calculateLabelsOfCentralCells(self.distanceOfCellsToCenter, self.centerRadius)

    def loadGeometricData(self, geometryFilename, skipFooter=0):
        if isinstance(geometryFilename, (str, Path)):
            geometryData = pd.read_csv(geometryFilename, skipfooter=skipFooter, engine="python")
        else:
            geometryData = geometryFilename
        if len(geometryData) == 0:
            warnings.warn(f"The geometry data was initialized with no data in it from {geometryFilename}")
        return geometryData

    def calculateCenterPosition(self, centerCellLabels):
        centerPosition = np.zeros(3)
        isCenterCell = np.isin(self.geometryData.loc[:, "Label"].values, centerCellLabels)
        assert np.any(isCenterCell), "None of the labels: {} exist/s as a label in the supplied data in file {}.".format(centerCellLabels, self.geometryDataFilename)
        mostCentralCellPositions = self.geometryData.loc[isCenterCell, ["Center_X", "Center_Y", "Center_Z"]].values
        if mostCentralCellPositions.shape[0] > 1:
            for xyz in mostCentralCellPositions:
                centerPosition += xyz
            centerPosition /= mostCentralCellPositions.shape[0]
        else:
            centerPosition = mostCentralCellPositions
        return centerPosition

    def calculateDistanceToCenter(self, centerPosition):
        xyzPositions = self.geometryData.loc[:, ["Center_X", "Center_Y", "Center_Z"]].values
        distanceXyzToCenter = xyzPositions - centerPosition
        euclideanDistanceToCenter = np.linalg.norm(distanceXyzToCenter, axis=1)
        return euclideanDistanceToCenter

    def calculateLabelsOfCentralCells(self, distanceOfCellsToCenter, centerRadius):
        isLabelInCentralRegion = distanceOfCellsToCenter <= centerRadius
        centralCellLabels = self.geometryData.loc[isLabelInCentralRegion, "Label"]
        nonCentralCellLabels = self.geometryData.loc[np.invert(isLabelInCentralRegion), "Label"]
        return centralCellLabels, nonCentralCellLabels

    def CalculateDistances(self):
        shape = self.geometryData.shape
        for i in range(shape[0]-1):
            for j in range(i, shape[0]):
                self.calculateDistanceBetween()

    def GetCentralCells(self):
        return self.centralCells

    def GetNonCentralCells(self):
        return self.nonCentralCellLabels

    def SaveCentralCells(self, filename):
        pass

def main():
    centerCellLabels = [618, 467, 570]
    geometryFilename = "./Data/connectivityNetworks/areaP1T0.csv"
    geometryData = pd.read_csv(geometryFilename, skipfooter=4)
    print(geometryData.shape)
    # ----- ATTENTION last 4 lines are skipped, make sure this does not change with MorphoGraphX Version before applying script -----
    myCellInSAMCenterDecider = CellInSAMCenterDecider(geometryData, centerCellLabels, centerRadius=30)
    print(myCellInSAMCenterDecider.GetCentralCells().shape)

if __name__ == '__main__':
    main()
