import numpy as np

def findSharedPoints(pointsFrom1, pointsFrom2, tolerance: float = np.sqrt(5), returnAllSharedIndices: bool = False):
    sharedPoints, indicesOfSharedPoints = [], []
    for i, p1 in enumerate(pointsFrom1):
        # print(pointsFrom2 - p1)
        distanceToP1 = np.linalg.norm(pointsFrom2 - p1, axis=1)
        # print(np.min(distanceToP1))
        isShared = distanceToP1 <= tolerance
        if np.any(isShared):
            sharedPoints.append(p1)
            if returnAllSharedIndices:
                indicesOfSharedPoints.append(np.where(isShared)[0])
            else:
                indicesOfSharedPoints.append(np.where(isShared)[0][0])
    return sharedPoints, indicesOfSharedPoints
