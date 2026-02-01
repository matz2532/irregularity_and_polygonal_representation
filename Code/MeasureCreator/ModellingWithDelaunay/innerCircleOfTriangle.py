import matplotlib.pyplot as plt
import numpy as np

from shapely import is_empty
from shapely.geometry import LineString, LinearRing

def pointsAlongCircle(radius, numberOfPoints):
    anglesOfPoints = np.linspace(0, 2*np.pi, numberOfPoints, endpoint=False)
    x = radius * np.sin(anglesOfPoints)
    y = radius * np.cos(anglesOfPoints)
    return np.concatenate([x,y]).reshape(2, numberOfPoints).T
def pointsAlongCircleWithPoint(centerPoint, radius, numberOfPoints):
    circle = pointsAlongCircle(radius, numberOfPoints)
    assert len(centerPoint) == 2, f"The Center point should be of length 2, but is {centerPoint}"
    circle += np.array(centerPoint)
    return circle
def plotRing(poins, ax, plotKwargs={}):
    ax.plot(np.concatenate([poins[:, 0], [poins[0, 0]]]), np.concatenate([poins[:, 1], [poins[0, 1]]]), **plotKwargs)
def plotCircleOf(centerPoint, radius, ax, numberOfPoints, circleKwargs={}):
    circle = pointsAlongCircleWithPoint(centerPoint, radius, numberOfPoints)
    plotRing(circle, ax, **circleKwargs)
    return circle

def pointArrayFromGeoms(shape):
    # .coords could be better for conversion
    return np.array([[pt.x, pt.y] for pt in shape.geoms])

def innerLine(triangleCornerPoints: np.ndarray, cornerIndex: int, ax=None, numberOfPointsForCircle=120):
    assert len(triangleCornerPoints) == 3, f"The triangle should have 3 points (length {len(triangleCornerPoints)} != 3) with points: {triangleCornerPoints}"
    if type(triangleCornerPoints) != np.ndarray:
        triangleCornerPoints = np.array(triangleCornerPoints)
    assert cornerIndex >= 0 and cornerIndex <= 2, "The index is out of range, should be between 0 - 2, but is {cornerIndex}"
    radius = 0.5 * np.min([np.linalg.norm(triangleCornerPoints[0] - triangleCornerPoints[1]),
                           np.linalg.norm(triangleCornerPoints[1] - triangleCornerPoints[2]),
                           np.linalg.norm(triangleCornerPoints[2] - triangleCornerPoints[0])])
    cornerPoint = triangleCornerPoints[cornerIndex, :]
    pointsOfCircleAroundCorner = pointsAlongCircleWithPoint(cornerPoint, radius, numberOfPointsForCircle)
    nextIndexPoint = cornerIndex + 1
    if nextIndexPoint >= len(triangleCornerPoints):
        nextIndexPoint -= len(triangleCornerPoints)
    previousIndexPoint = cornerIndex - 1
    cornerCircleRing = LinearRing(pointsOfCircleAroundCorner)
    firstIntersectionPoint = LineString([triangleCornerPoints[previousIndexPoint, :], cornerPoint]).intersection(cornerCircleRing)
    secondIntersectionPoint = LineString([cornerPoint, triangleCornerPoints[nextIndexPoint, :]]).intersection(cornerCircleRing)
    intersectionPoints = np.array([[firstIntersectionPoint.x, firstIntersectionPoint.y],
                          [secondIntersectionPoint.x, secondIntersectionPoint.y]])
    intersectionCircle_1 = pointsAlongCircleWithPoint(intersectionPoints[0], radius, numberOfPointsForCircle)
    intersectionCircle_2 = pointsAlongCircleWithPoint(intersectionPoints[1], radius, numberOfPointsForCircle)
    ringsIntersect = LinearRing(intersectionCircle_1).intersection(LinearRing(intersectionCircle_2))
    if ringsIntersect.is_empty:
        # add visual indication for problematic intersect
        plt.close()
        fig, ax = plt.subplots(figsize=(6,6), constrained_layout=True)
        ax.scatter(triangleCornerPoints[:, 0], triangleCornerPoints[:, 1])
        plotRing(intersectionCircle_1, ax)
        plotRing(intersectionCircle_2, ax)
        plt.show()
    pointsForLinearSection = pointArrayFromGeoms(ringsIntersect)
    if ax is not None:
        plotRing(pointsOfCircleAroundCorner, ax)
        ax.scatter(intersectionPoints[:, 0], intersectionPoints[:, 1])
        plotRing(intersectionCircle_1, ax)
        plotRing(intersectionCircle_1, ax)
        ax.plot(pointsForLinearSection[:, 0], pointsForLinearSection[:, 1])
    m, n = np.polyfit(pointsForLinearSection[:, 0], pointsForLinearSection[:, 1], 1)
    return {"m": m, "n": n}

def intersectFrom(mn1Dict, mn2Dict, mKey="m", nKey="n"):
    m1 = mn1Dict[mKey]
    n1 = mn1Dict[nKey]
    m2 = mn2Dict[mKey]
    n2 = mn2Dict[nKey]
    x = (n2 - n1) / (m1 - m2)
    y = m1 * x + n1
    return [x, y]

def calcInnerCircleOfTriangle(triangleCorners, ax=None):
    mn1Dict = innerLine(triangleCorners, 0, ax)
    mn2Dict = innerLine(triangleCorners, 1, ax)
    intersect = intersectFrom(mn1Dict, mn2Dict)
    if ax is not None:
        ax.scatter(intersect[0], intersect[1])
    return intersect

def mainTestingTriangleCentroidError():
    from shapely.geometry import Polygon
    # correct Centroid of triangle
    triangleCorners = [[-5.6, 28.8],
                       [5.9, 23.1],
                       [-3.5, 6]]
    wrongCentroid = [-0.8, 18.8]
    # problematic triangles centroid
    triangleCorners = [[-9.75, -18.5],
                       [-4.9, -23],
                       [-2.9, -39.8]]
    wrongCentroid = [-5, -27]

    addGeometricEstimationOfInnerCircle = True
    # try to find out whether order of triangles is importent when calculating centroid
    trianglePolygon_1 = Polygon(triangleCorners)
    xyOfCentroid_1 = trianglePolygon_1.centroid
    trianglePolygon_2 = Polygon(np.array([triangleCorners[1], triangleCorners[0], triangleCorners[2]]))
    xyOfCentroid_2 = trianglePolygon_2.centroid
    trianglePolygon_3 = Polygon(np.array([triangleCorners[2], triangleCorners[1], triangleCorners[0]]))
    xyOfCentroid_3 = trianglePolygon_3.centroid

    fig, ax = plt.subplots(figsize=(6,6), constrained_layout=True)
    triangleCornersForPlotting = np.concatenate([triangleCorners, [triangleCorners[0]]], axis=0)
    ax.plot(triangleCornersForPlotting[:, 0], triangleCornersForPlotting[:, 1])
    ax.scatter(wrongCentroid[0], wrongCentroid[1], label="original wrong")
    ax.scatter(xyOfCentroid_1.x, xyOfCentroid_1.y, label="1")
    ax.scatter(xyOfCentroid_2.x, xyOfCentroid_2.y, label="2")
    ax.scatter(xyOfCentroid_3.x, xyOfCentroid_3.y, label="3")
    meanCornerPoint = np.mean(triangleCornersForPlotting, axis=0)
    ax.scatter(meanCornerPoint[0], meanCornerPoint[1], label="mean")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    newLim = (np.min([xlim[0], ylim[0]]), np.max([xlim[1], ylim[1]]))
    ax.set_xlim(newLim)
    ax.set_ylim(newLim)

    if addGeometricEstimationOfInnerCircle:
        mn1Dict = innerLine(triangleCorners, 0, ax)
        mn2Dict = innerLine(triangleCorners, 1, ax)
        mn3Dict = innerLine(triangleCorners, 2, ax)
        intersect = intersectFrom(mn2Dict, mn3Dict)
        plt.plot([intersect[0], triangleCorners[2][0]], [intersect[1], triangleCorners[2][1]])
        plt.scatter(*intersect)

    plt.legend()
    plt.show()

if __name__ == "__main__":
    mainTestingTriangleCentroidError()
