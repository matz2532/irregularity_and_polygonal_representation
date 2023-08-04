import matplotlib.axes
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.color
import skimage.transform
import sys
import time
import warnings

sys.path.insert(0, "./Code/DataStructures/")

from FolderContent import FolderContent
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plottingUtils import createColorMapper
from pyntcloud import PyntCloud
from scipy.spatial.transform import Rotation

class PatchCreator (object):

    colorBar=None
    defaultFaceColor="#00000000"
    isThreeDimensional=True
    patchAlpha=1
    patchCollection=None
    showEveryXthOutlinePoint=1
    tissueName=None
    verbosity=1

    def __init__(self):
        pass

    def GetColorBar(self):
        return self.colorBar

    def AddPatchesToAxis(self, ax, patchCollection=None):
        if patchCollection is None:
            patchCollection = self.patchCollection
        if not patchCollection is None:
            return ax.add_collection(patchCollection)
        else:
            if self.verbosity > 0:
                print(f"The patches could not be added to the axis as both {patchCollection=} and {self.patchCollection=} are None.")
            return None

    def PlotPatchesFromOutlineDictOn(self, ax, outlineDict, **kwargs):
        self.patchCollection = self.createPatchesFromOutlineDict(outlineDict, **kwargs)
        return self.AddPatchesToAxis(ax)

    def Plot3DPatchesFromOutlineDictOn(self, outlineDict, ax=None, colorMapper=None, colorMap=None, measureData=None, plotOutlines=True,
                                       polygonalOutlineDict=None, faceColorDict=None, isThreeDimensional=None,
                                       showColorBar=True, alphaAsFloat=0.5, colorMapValueRange=None, scaleBarSize=None, scaleBarOffset=None, markCells=None, tissueName=None,
                                       axesParameter=None, showPlot=False, title=None, colorBarLabel="polygonal area [µm²]", getCellIdByKeyStroke=True,
                                       backgroundFilename=None, is3DBackground=False, backgroundImageOffset: np.ndarray = None,
                                       outlineLineWidth=1.5, polygonOutlineLineWidth=1, ignoreAxis=True, **kwargs):
        if not isThreeDimensional is None:
            self.isThreeDimensional = isThreeDimensional
        if not tissueName is None:
            self.tissueName = tissueName
        if ax is None:
            if self.isThreeDimensional:
                projection = "3d"
            else:
                projection = None
            fig = plt.figure()
            self.ax = fig.add_subplot(projection=projection)
            self.cellLabels = list(outlineDict.keys())
            self.cellPositions = np.array([np.mean(contourPoints, axis=0) for contourPoints in outlineDict.values()])
            self.getCellIdByKeyStroke = getCellIdByKeyStroke
            fig.canvas.mpl_connect("key_press_event", self.onPress)
        else:
            self.ax = ax
        if measureData is None:
            showColorBar = False
        else:
            if colorMapper is None:
                colorMapper = self.createColorMapper(colorMap=colorMap, alphaAsFloat=alphaAsFloat,
                                                     colorMapValueRange=colorMapValueRange,
                                                     measureData=measureData, keysToSelect=list(outlineDict.keys()))
        if not faceColorDict is None:
            defaultFaceColor = faceColorDict
            colorMapperForFaceColor = False
        else:
            defaultFaceColor = (0,0,0,0)
            colorMapperForFaceColor = True

        if plotOutlines:
            allPatches = self.create3DPatchesFromOutlines(outlineDict, colorMapper=colorMapper, measureData=measureData, defaultLineWidth=outlineLineWidth,
                                                          defaultFaceColor=defaultFaceColor, colorMapperForFaceColor=colorMapperForFaceColor)
            if self.isThreeDimensional:
                for pc in allPatches:
                    self.ax.add_collection3d(pc)
            else:
                for pc in allPatches:
                    self.ax.add_collection(pc)
        if not polygonalOutlineDict is None:
            polygonalBackgroundColor = self.create3DPatchesFromOutlines(polygonalOutlineDict, defaultEdgeColor="green", colorMapperForFaceColor=False, defaultLineWidth=polygonOutlineLineWidth)
            # polygonalEdgeColor = self.create3DPatchesFromOutlines(polygonalOutlineDict, colorMapper=colorMapper, measureData=measureData, colorMapperForFaceColor=False, colorMapperForEdgeColor=True, defaultLineWidth=1.5)
            for background in polygonalBackgroundColor:
                if self.isThreeDimensional:
                    self.ax.add_collection3d(background)
                else:
                    self.ax.add_collection(background)
            # for pc in polygonalEdgeColor:
            #     self.ax.add_collection3d(pc)
        if not markCells is None:
            self.drawMarkerAtCenterOfOutlines(self.ax, markCells, outlineDict)
        if axesParameter is None:
            self.add3DLimits(outlineDict, self.ax)
            if self.isThreeDimensional:
                if outlineDict[list(outlineDict.keys())[0]].shape[1] > 2:
                    self.ax.view_init(elev=50, azim=120)
                else:
                    self.ax.view_init(elev=90, azim=-90)
        else:
            self.initVisualisationFrom(axesParameter, self.ax)
        if not scaleBarSize is None:
            if self.isThreeDimensional:
                self.drawScaleBar3D(scaleBarSize, scaleBarOffset=scaleBarOffset)
            else:
                if scaleBarOffset is None:
                    xLim, yLim = self.ax.get_xlim(), self.ax.get_ylim()
                    xPos = xLim[0] + 0.15 * (xLim[1] - xLim[0])
                    yPos = yLim[0] + 0.05 * (yLim[1] - yLim[0])
                    scaleBarOffset = np.array([xPos, yPos])
                self.drawScaleBar2D(scaleBarSize, scaleBarOffset=scaleBarOffset, scaleBarColor="black")
        if not backgroundFilename is None:
            if self.isThreeDimensional:
                if is3DBackground:
                    self.ax.set_facecolor("#000000")
                    self.add3DSurfaceFromPlyFile(self.ax, backgroundFilename, offset=backgroundImageOffset)
                else:
                    pass # self.add2DBackground(self.ax, backgroundFilename, outlineDict, offset=backgroundImageOffset)
            else:
                backgroundImage = FolderContent({}).loadFile(backgroundFilename)
                backgroundImage = backgroundImage.T
                self.ax.imshow(backgroundImage, cmap="gray")
        if ignoreAxis:
            self.ax.axis("off")
        else:
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('y')
            if self.isThreeDimensional:
                self.ax.set_zlabel('z')
        plt.tight_layout()
        if not title is None:
            self.ax.set_title(title, y=1.11, size=35)
        if showColorBar:
            cbar = plt.colorbar(colorMapper)
            # cbar.ax.get_yaxis().labelpad = 25
            if not colorBarLabel is None:
                cbar.set_label(colorBarLabel, rotation=90)
        if showPlot:
            plt.show()
        return self.ax

    def createColorMapper(self, colorMap: colors.ListedColormap = None, alphaAsFloat: float = 1,
                          colorMapValueRange: tuple = None,
                          measureData: dict or pd.DataFrame = None,
                          keysToSelect: list = None, valueColIdx: int = 1):
        if colorMap is None:
            colorMap = plt.get_cmap("plasma")
        myColorMap = colorMap(np.arange(colorMap.N))
        myColorMap[:, -1] = alphaAsFloat
        myColorMap = colors.ListedColormap(myColorMap)
        assert not colorMapValueRange is None or isinstance(measureData, (dict, pd.DataFrame)), f"You need to either provide the color map value range or the measure data to calculate the range as dict or DataFrame {type(measureData)=}"
        if colorMapValueRange is None:
            if isinstance(measureData, dict):
                if keysToSelect is None:
                    allPotentialValues = [data for data in measureData.values()]
                else:
                    allPotentialValues = [measureData[cell] for cell in keysToSelect]
            else:
                allPotentialValues = measureData.iloc[:, valueColIdx]
            colorMapValueRange = [np.min(allPotentialValues), np.max(allPotentialValues)]
        colorMapper = createColorMapper(valueRange=colorMapValueRange, colorMap=myColorMap)
        return colorMapper

    def onPress(self, event, decimal=2, printOutAxesParameterEveryChange: bool = False):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        zlim = self.ax.get_zlim()
        azim = self.ax.azim
        elev = self.ax.elev
        axesParameter = {"xlim":np.round(xlim, decimal).tolist(), "ylim":np.round(ylim, decimal).tolist(), "zlim":np.round(zlim, decimal).tolist(), "azim":np.round(azim, decimal), "elev":np.round(elev, decimal)}

        if printOutAxesParameterEveryChange:
            print(f'"{self.tissueName}": {axesParameter},')
        if self.getCellIdByKeyStroke:
            mousePosition = self.pick_handler(event, self.ax)
            if self.cellPositions.shape[1] == 2:
                mousePosition = mousePosition[:2]
            distToMouse = np.linalg.norm(self.cellPositions-mousePosition, axis=1)
            closestCellsIdx = np.argmin(distToMouse)
            print(self.cellLabels[closestCellsIdx], ", ")

    def pick_handler(self, event, ax):
        # adapted from user95209 of https://stackoverflow.com/questions/30674526/matplotlib-getting-coordinates-in-3d-plots-by-a-mouseevent#43917457
        if ax.M is None:
            return {}

        xd, yd = event.xdata, event.ydata
        p = (xd, yd)
        edges = self.ax.tunit_edges()
        ldists = [(self.line2d_seg_dist(p0, p1, p), i) for \
                  i, (p0, p1) in enumerate(edges)]
        ldists.sort()

        # nearest edge
        edgei = ldists[0][1]

        p0, p1 = edges[edgei]

        # scale the z value to match
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        d0 = np.hypot(x0 - xd, y0 - yd)
        d1 = np.hypot(x1 - xd, y1 - yd)
        dt = d0 + d1
        z = d1 / dt * z0 + d0 / dt * z1

        x, y, z = mplot3d.proj3d.inv_transform(xd, yd, z, ax.M)
        return np.array([x, y, z])

    def line2d_seg_dist(self, p1, p2, p0):
        """distance(s) from line defined by p1 - p2 to point(s) p0

        p0[0] = x(s)
        p0[1] = y(s)

        intersection point p = p1 + u*(p2-p1)
        and intersection point lies within segment if u is between 0 and 1

        suggested to vendor (original: mpl_toolkits.mplot3d.proj3d.line2d_seg_dist) function
        code from ImportanceOfBeingErnest of https://stackoverflow.com/questions/58065380/with-what-to-replace-the-deprecated-line2d-seg-dist-function-of-matplotlib
        """

        x21 = p2[0] - p1[0]
        y21 = p2[1] - p1[1]
        x01 = np.asarray(p0[0]) - p1[0]
        y01 = np.asarray(p0[1]) - p1[1]

        u = (x01 * x21 + y01 * y21) / (x21 ** 2 + y21 ** 2)
        u = np.clip(u, 0, 1)
        d = np.hypot(x01 - u * x21, y01 - u * y21)

        return d

    def drawMarkerAtCenterOfOutlines(self, ax: matplotlib.axes.Axes, labelsOfOutlinesToMark: list, outlineOfLabels: dict):
        cellCentersToMark = []
        for cellLabel in labelsOfOutlinesToMark:
            if cellLabel in outlineOfLabels:
                cellContour = outlineOfLabels[cellLabel]
                roughGeometricCenter = np.mean(cellContour, axis=0)
                cellCentersToMark.append(roughGeometricCenter)
            else:
                print(f"The {cellLabel} is not present as an outline, while it should have been marked. Only {np.sort(list(outlineOfLabels.keys()))} are present")
        if len(cellCentersToMark) > 0:
            cellCentersToMark = np.array(cellCentersToMark)
            ax.scatter(cellCentersToMark[:, 0], cellCentersToMark[:, 1], cellCentersToMark[:, 2], c="black", marker="*")

    def drawOutlinesIn3D(self, outlineDict, ax):
        for cellLabel, currentContour in outlineDict.items():
            x = np.concatenate([currentContour[:, 0], [currentContour[0, 0]]])
            y = np.concatenate([currentContour[:, 1], [currentContour[0, 1]]])
            shapeOfContour = currentContour.shape
            if shapeOfContour[1] > 2:
                z = np.concatenate([currentContour[:, 2], [currentContour[0, 2]]])
            else:
                z = np.zeros(shapeOfContour[0] + 1)
            ax.plot(x, y, z, c="black", linewidth=3)

    def create3DPatchesFromOutlines(self, outlineDict, colorMapper=None, measureData=None,
                                    defaultFaceColor=(0,0,0,0), colorMapperForFaceColor=True,
                                    defaultEdgeColor="black", colorMapperForEdgeColor=False, defaultLineWidth=1):
        allPatches = []
        for cellLabel, currentContour in outlineDict.items():
            shapeOfContour = currentContour.shape
            if self.isThreeDimensional:
                if shapeOfContour[1] == 2:
                    currentContour = np.concatenate([currentContour, np.zeros(shapeOfContour[0]).reshape(shapeOfContour[0], 1)], axis=1)
            else:
                if shapeOfContour[1] == 3:
                    currentContour = currentContour[:, :2]
            if type(defaultFaceColor) == dict:
                if cellLabel in defaultFaceColor:
                    faceColor = defaultFaceColor[cellLabel]
                else:
                    print(f"The {cellLabel=} was not present in the defaultFaceColor dict, only the following labels are present {np.sort(list(defaultFaceColor.keys())).tolist()}")
            else:
                faceColor = defaultFaceColor
            edgeColor = defaultEdgeColor
            if not colorMapper is None and (colorMapperForFaceColor or colorMapperForEdgeColor):
                if not measureData is None:
                    try:
                        if isinstance(measureData, dict):
                            value = measureData[cellLabel]
                        else:
                            cellsRow = measureData.loc[measureData.iloc[:, 0] == cellLabel]
                            value = cellsRow.iloc[:, 1].to_list()[0]
                    except KeyError as e:
                        if isinstance(measureData, dict):
                            presentCellLabels = list(np.sort(list(measureData.keys())))
                        else:
                            presentCellLabels = list(np.sort(cellsRow.iloc[:, 0]))
                        print(f"Plotting the {self.tissueName=} the following key was missing {cellLabel=} a value and was therefore skipped, {presentCellLabels=}")
                if colorMapperForFaceColor:
                    faceColor = colorMapper.to_rgba(value)
                if colorMapperForEdgeColor:
                    edgeColor = colorMapper.to_rgba(value)
            if self.isThreeDimensional:
                pc = Poly3DCollection([currentContour], facecolors=[faceColor], edgecolors=[edgeColor], linewidths=[defaultLineWidth])
            else:
                pc = PatchCollection([Polygon(currentContour)], facecolors=[faceColor], edgecolors=[edgeColor], linewidths=[defaultLineWidth])
            allPatches.append(pc)
        return allPatches

    def add3DLimits(self, outlineDict, ax, additionalOffset=25):
        allMins, allMaxs = [], []
        for cellLabel, currentContour in outlineDict.items():
            min = np.min(currentContour, axis=0)
            max = np.max(currentContour, axis=0)
            allMins.append(min)
            allMaxs.append(max)
        overAllMin = np.min(allMins, axis=0) - additionalOffset
        overAllMax = np.max(allMaxs, axis=0) + additionalOffset
        limits = list(zip(overAllMin, overAllMax))
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        if self.isThreeDimensional:
            if len(limits) > 2:
                ax.set_zlim(limits[2])
            else:
                ax.set_zlim([0, 5])

    def initVisualisationFrom(self, axesParameter, ax):
        if "xlim" in axesParameter:
            ax.set_xlim(axesParameter["xlim"])
        if "ylim" in axesParameter:
            ax.set_ylim(axesParameter["ylim"])
        if self.isThreeDimensional:
            if "zlim" in axesParameter:
                ax.set_zlim(axesParameter["zlim"])
            if "elev" in axesParameter and "azim" in axesParameter:
                ax.view_init(elev=axesParameter["elev"], azim=axesParameter["azim"])

    def drawScaleBar3D(self, scaleBarSize, scaleBarOffset: np.ndarray = None, scaleBarColor: str or list = "grey"):
        roll, elevation, azimuth = self.ax.roll, self.ax.elev, self.ax.azim
        if -45 <= azimuth <= 45:
            azimuthToRotateBy = - azimuth
        elif azimuth > 45 or azimuth < - 45:
            azimuthToRotateBy = azimuth - 90
        rotationVector = [-elevation, azimuthToRotateBy]
        r = Rotation.from_euler("xz", rotationVector, degrees=True)
        scaleBarPositions = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        scaleBarPositions = r.apply(scaleBarPositions)
        scaleBarPositions[1, :] *= scaleBarSize / np.linalg.norm(scaleBarPositions[1, :])
        scaleBarPositions -= scaleBarPositions[1, :] / 2
        if scaleBarOffset is None:
            xLim, yLim, zLim = self.ax.get_xlim(), self.ax.get_ylim(), self.ax.get_zlim()
            offsetRatio = np.array([0.01, 0.01, 0.01])
            leftFrontOffset = np.array([xLim[0], yLim[1], zLim[0]])
            baseOffset = np.array([xLim[1] - xLim[0], yLim[1] - yLim[0], zLim[1] - zLim[0]])
            baseOffset *= offsetRatio
            scaleBarOffset = leftFrontOffset - baseOffset
        scaleBarPositions += scaleBarOffset
        self.ax.plot(scaleBarPositions[:, 0], scaleBarPositions[:, 1], scaleBarPositions[:, 2], color=scaleBarColor, linewidth=4)

    def drawScaleBar2D(self, scaleBarSize, scaleBarOffset: np.ndarray = None, scaleBarColor: str or list = "grey"):
        scaleBarPositions = np.array([[0, 0], [1, 0]], dtype=float)
        scaleBarPositions[1, :] *= scaleBarSize / np.linalg.norm(scaleBarPositions[1, :])
        scaleBarPositions -= scaleBarPositions[1, :] / 2
        if not scaleBarOffset is None:
            scaleBarPositions += scaleBarOffset
        self.ax.plot(scaleBarPositions[:, 0], scaleBarPositions[:, 1], color=scaleBarColor, linewidth=2)

    def add2DBackground(self, ax, backgroundFilename, outlinesOfCells, offset=None, downScaleImageBy: int = 30):
        backgroundImage = FolderContent({}).loadFile(backgroundFilename) # , as_gray=True
        xLim, yLim, zLim = self.ax.get_xlim(), self.ax.get_ylim(), self.ax.get_zlim()
        imageShape = backgroundImage.shape
        if len(imageShape) == 3: # images is rgb
            backgroundImage = skimage.color.rgb2gray(backgroundImage)
        # maxSize = 0.221 * np.array(backgroundImage.shape)
        # print(maxSize)
        backgroundImage = skimage.transform.resize(backgroundImage,
                                                   (backgroundImage.shape[0] // downScaleImageBy, backgroundImage.shape[1] // downScaleImageBy),
                                                   anti_aliasing=True)
        backgroundImage = backgroundImage.T
        backgroundImage = np.flip(backgroundImage, axis=1)
        imageShape = backgroundImage.shape
        first = True
        for contour in outlinesOfCells.values():
            min = np.min(contour, axis=0)
            max = np.max(contour, axis=0)
            if first:
                xLim = [min[0], max[0]]
                yLim = [min[1], max[1]]
                first = False
            else:
                if min[0] < xLim[0]:
                    xLim[0] = min[0]
                if min[1] < yLim[0]:
                    yLim[0] = min[1]
                if max[0] > xLim[1]:
                    xLim[1] = max[0]
                if max[1] > yLim[1]:
                    yLim[1] = max[1]
        print(xLim, yLim)
        # xx, yy = np.meshgrid(np.linspace(*xLim, imageShape[1]), np.linspace(*yLim, imageShape[0]))
        # xx, yy = np.meshgrid(np.linspace(-315, 314.5, imageShape[1]), np.linspace(-471.5, 417.5, imageShape[0]))
        xx, yy = np.meshgrid(np.linspace(0, 629.5, imageShape[1]), np.linspace(0, 945, imageShape[0]))
        # ax.contourf(xx, yy, backgroundImage, 255, zdir='z', offset=0, cmap="Greys_r")
        ax.contourf(xx, yy, backgroundImage, zdir='z', offset=-0.1, cmap="Greys_r")
        print("after")

    def add3DSurfaceFromPlyFile(self, ax, backgroundPlyFilename, positionColumns=["x", "y", "z"], offset=np.array([0, 0, -0.25])):
        pointCloud = PyntCloud.from_file(backgroundPlyFilename)
        mesh = pointCloud.mesh
        points = pointCloud.points
        if not offset is None:
            points[positionColumns] += offset
        self.addSignalOfSurfaceFrom(ax, mesh, points, positionColumns=positionColumns)

    def addSignalOfSurfaceFrom(self, ax, meshDf, pointsDf, alpha=0.8, signalColumn = "signal",
                               positionColumns = ["x", "y", "z"], positionIndexColumns = ["v1", "v2", "v3"],
                               colorMap=None, printOutTimeAtIntervals: int = 5):
        pointColumns = pointsDf.columns
        signalColIdx = np.where(pointColumns == signalColumn)[0][0]
        positionColIdx = np.where(np.isin(pointColumns, positionColumns))[0]
        if colorMap is None:
            colorMap = plt.get_cmap("Greys_r")
        colorMapper = PatchCreator().createColorMapper(colorMap, measureData=pointsDf, valueColIdx=signalColIdx, alphaAsFloat=alpha)
        allFacePositions, allFaceColors, allLineWidths = [], [], []
        s = time.time()
        if printOutTimeAtIntervals:
            fivePercentStops = len(meshDf) // printOutTimeAtIntervals
        for i, vertexIndicesRow in meshDf.iterrows():
            if printOutTimeAtIntervals:
                if i % fivePercentStops == 0 and i > 0:
                    t = time.time() - s
                    print(f"{i//fivePercentStops}/{printOutTimeAtIntervals} of {len(meshDf)}, {np.round(t, 3)}s, {np.round(t / 60, 3)}min, {np.round(t / 3600, 3)}h")
            vertexIndices = vertexIndicesRow[positionIndexColumns].values.astype(int)
            facePositions = pointsDf.iloc[vertexIndices, positionColIdx].to_numpy()
            allFacePositions.append(facePositions)
            meanSignalOfFace = pointsDf.iloc[vertexIndices, signalColIdx].mean()
            faceColor = colorMapper.to_rgba(meanSignalOfFace)
            allFaceColors.append(faceColor)
            allLineWidths.append(0)
        pc = Poly3DCollection(allFacePositions, facecolors=allFaceColors, linewidths=allLineWidths)
        ax.add_collection3d(pc)

    def createPatchesFromOutlineDict(self, outlineDict, faceColor=None,
                                     edgecolor="#ffffff", alpha=None,
                                     flipOutlineCoordinates=True,
                                     removeColorBar=None, colorMapper=None, **kwargs):
        useMapper = False
        if faceColor is None:
            faceColor = self.defaultFaceColor
        elif type(faceColor) == dict:
            useMapper = True
            if colorMapper is None:
                values = list(faceColor.values())
                colorMapper = createColorMapper(valueRange=[np.min(values), np.max(values)])
        patches = []
        patchesFaceColors, patchesEdgeColors = [], []
        if isinstance(faceColor, list):
            patchesFaceColors = faceColor
        for id, contour in outlineDict.items():
            if useMapper:
                patchesFaceColors.append(colorMapper.to_rgba(faceColor[id]))
            else:
                if not isinstance(faceColor, list):
                    patchesFaceColors.append(faceColor)
            patchesEdgeColors.append(edgecolor)
            if flipOutlineCoordinates:
                contour = np.flip(contour, axis=1)
            if not contour is None:
                patches.append(Polygon(contour[::self.showEveryXthOutlinePoint]))
        if len(patches) == 0:
            if self.verbosity > 0:
                warnings.warn("The patch list was empty!")
        edgecolors = None if edgecolor is None else patchesEdgeColors
        patchCollection = PatchCollection(patches, facecolors=patchesFaceColors, edgecolors=edgecolors, alpha=alpha, **kwargs)
        if useMapper:
            if not removeColorBar is None:
                removeColorBar.remove()
            self.colorBar = plt.colorbar(colorMapper)
        return patchCollection

