import argparse
import numpy as np
import pandas as pd
import re, sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

class ConvertTextToLabels (object):

    extractedLabelsFilename=None

    def __init__(self, textOrFilename, filenameToSave=None, allLabelsFilename=None,
                filenameToSaveControl=None, howToJoinLabels=", ", onlyUnique=True,
                isText=False, dontSave=True, saveOnlyHeatMapSelection=False):
        self.howToJoinLabels = howToJoinLabels
        self.allLabelsFilename = allLabelsFilename
        if isText:
            self.lines = textOrFilename
        else:
            self.lines = self.convertTextOrFilenameToLines(textOrFilename)
        self.labels = self.calculateLabels(self.lines)
        if not dontSave:
            if not filenameToSave is None:
                self.extractedLabelsFilename = filenameToSave
            else:
                self.extractedLabelsFilename = textOrFilename[:-4] + "_final.txt"
            self.saveLabels(self.extractedLabelsFilename, howToJoinLabels, onlyUnique)
        if not allLabelsFilename is None:
            if filenameToSaveControl is None and not dontSave:
                folder = Path(textOrFilename).parent
                filename = Path(textOrFilename).stem + "_control.csv"
                self.filenameToSaveControl = Path(folder).joinpath(filename)
            else:
                self.filenameToSaveControl = filenameToSaveControl
            self.controlLabelPosition(self.labels, self.filenameToSaveControl, allLabelsFilename,
                                      allLabelsAsHeatMapData=True, save=not dontSave)
            if saveOnlyHeatMapSelection:
                folder = Path(textOrFilename).parent
                filename = Path(self.allLabelsFilename).stem + "_selectedHeatMap.csv"
                filenameToSave = Path(folder).joinpath(filename)
                print(filenameToSave)
                self.saveOnlyHeatMapOfLabels(self.labels, self.table, filenameToSave)

    def GetExtractedLabelsFilename(self):
        return self.extractedLabelsFilename

    def convertTextOrFilenameToLines(self, textOrFilename):
        assert type(textOrFilename) is str, "You need to supply a string of text or a filename."
        splitByDot = textOrFilename.split(".")
        if len(splitByDot) > 1:
            if len(splitByDot) > 2:
                print("You may not passing a valid filename as you have more than one '.' .")
            file = open(textOrFilename, "r")
            textOrFilename = file.readlines()
            file.close()
            for i in range(len(textOrFilename)):
                textOrFilename[i] = textOrFilename[i].rstrip("\n")
        else:
            textOrFilename = textOrFilename.split("\n")
        return textOrFilename

    def calculateLabels(self, lines):
        labels = []
        for i in range(len(lines)):
            foundString = re.findall("\d+", lines[i])
            if len(foundString) > 0:
                if len(foundString) == 1:
                    labels.append(foundString[0])
                else:
                    warningMsg = """More than one continious number was found in line {}: {}
Only the first number was added.""".format(i+1, foundString)
                    print(warningMsg)
                    labels.append(foundString[0])
        if len(labels) == 0:
            print("No label was found as there were no integers found.")
        return labels

    def saveLabels(self, filenameToSave, howToJoinLabels, onlyUnique=True):
        labels = self.labels
        if onlyUnique is True:
            labels = np.unique(self.labels)
        joinedLabels = howToJoinLabels.join(labels)
        file = open(filenameToSave, "w")
        file.write(joinedLabels)
        file.close()

    def controlLabelPosition(self, labels, saveToFilename, allLabels, allLabelsAsHeatMapData=True, save=False):
        labels = np.unique(labels).astype(int)
        if allLabelsAsHeatMapData is True:
            table = pd.read_csv(allLabels, skipfooter=4, engine="python")
            self.table = table.copy()
        else:
            print("other measures arent yet implemented. Set allLabelsAsHeatMapData to True")
            sys.exit()
        isLabelSelected = np.isin(table["Label"], labels)
        table["Value"] = np.asarray(isLabelSelected).astype(int)
        if save:
            table.to_csv(saveToFilename, index=False)
            if allLabelsAsHeatMapData is True:
                lastLines = self.extractLastLines(allLabels, extractFooter=4)
                self.appendLinesToFile(lastLines, saveToFilename)

    def extractLastLines(self, fileToOpen, extractFooter=0):
        lastLines = ""
        file = open(fileToOpen, "r")
        allLines = file.readlines()
        file.close()
        if int(extractFooter) > 0:
            if len(allLines) >= extractFooter:
                lastLines = allLines[-extractFooter:]
                lastLines = "".join(lastLines)
                # do I need to change the range or the mesh number?
            else:
                print("The number of footer lines to extract ({}) was larger than the number of lines in the file {}.".format(extractFooter, fileToOpen))
        return lastLines

    def appendLinesToFile(self, lastLines, saveToFilename):
        file = open(saveToFilename, "a")
        file.write(lastLines)
        file.close()

    def saveOnlyHeatMapOfLabels(self, labels, table, filenameToSave):
        allLabels = table["Label"]
        labels = np.asarray(labels).astype(int)
        isLabelInTable = np.isin(labels, allLabels)
        assert np.all(isLabelInTable), "While getting the label positons, the labels {} are not present in the heat map file.".format(np.asarray(labels)[isLabelInTable])
        idxOfLabels = np.where(np.isin(allLabels, labels))[0]
        selectionOfTable = table.iloc[idxOfLabels, :]
        selectionOfTable.to_csv(filenameToSave, index=False)

    def GetLabels(self, onlyUnique=True):
        labels = np.asarray(self.labels).astype(int)
        if onlyUnique is True:
            indexes = np.unique(labels, return_index=True)[1]
            labels = [labels[index] for index in sorted(indexes)]
        return labels

    def GetLabelsPositions(self, positionColumnNames=["Center_X", "Center_Y", "Center_Z"]):
        labels = self.GetLabels(onlyUnique=True)
        if not self.allLabelsFilename is None:
            allLabels = self.table["Label"]
            isLabelInTable = np.isin(labels, allLabels)
            assert np.all(isLabelInTable), "While getting the label positons, the labels {} are not present in the heat map file.".format(np.asarray(labels)[isLabelInTable])
            idxOfLabels = np.where(np.isin(allLabels, labels))[0]
            columnNames = self.table.columns
            ispositionColumnPresent = np.isin(positionColumnNames, columnNames)
            assert np.all(ispositionColumnPresent), "While getting the label positons, the positional columns {} are not present in the heat map file.".format(np.asarray(positionColumnNames)[ispositionColumnPresent])
            positionColumnIdx = np.where(np.isin(columnNames, positionColumnNames))[0]
            positions = self.table.iloc[idxOfLabels, positionColumnIdx].to_numpy()
            return positions

def extractSelectedCellsFor(tissueBaseFilename, loadFromBaseName="_peripheralCells.txt", geometricTableBaseName="_geometricData full.csv"):
    peripheralCellsTextFilename = tissueBaseFilename + loadFromBaseName
    geometricDataFilename = tissueBaseFilename + geometricTableBaseName
    ConvertTextToLabels(str(peripheralCellsTextFilename), allLabelsFilename=geometricDataFilename, dontSave=False)

def mainExtractPeripheralCellsForMultiple(baseFolder="Images/full cotyledons/WT/", allTissueReplicateIds=["20200220 WT S1", "20200221 WT S2", "20200221 WT S3", "20200221 WT S5"]):
    for tissueReplicateId in allTissueReplicateIds:
        peripheralCellsTextFilename = f"{baseFolder}{tissueReplicateId}/{tissueReplicateId}_peripheralCells.txt"
        geometricDataFilename = f"{baseFolder}{tissueReplicateId}/{tissueReplicateId}_geometricData full.csv"
        ConvertTextToLabels(peripheralCellsTextFilename, allLabelsFilename=geometricDataFilename, dontSave=False)

def mainCreateCellTypeOfProposedGuardCells(baseFolder="Images/", allTissueIdentifiers=[["first_leaf_LeGloanec2022", "01", "6DAI"],
                                                                                       ["first_leaf_LeGloanec2022", "02", "6DAI"],
                                                                                       ["first_leaf_LeGloanec2022", "03", "6DAI"]],
                                           cellTypeExtension="_CELL_TYPE.csv"):
    for tissueIdentifier in allTissueIdentifiers:
        genotype, tissueReplicateId, timePoint = tissueIdentifier
        baseFilename = str(Path(baseFolder).joinpath(genotype, tissueReplicateId, timePoint, f"{tissueReplicateId}_{timePoint}"))
        print(baseFilename)
        proposedGuardCellsTextFilename = f"{baseFilename}_proposed guard cells.txt"
        geometricDataFilename = f"{baseFilename}_geometricData.csv"
        cellTypeFilename = f"{baseFilename}{cellTypeExtension}"
        guardCells = ConvertTextToLabels(proposedGuardCellsTextFilename, dontSave=False).GetLabels()
        # default values for loading cell types from table
        stomataTypeId = 16
        otherCellId = 5
        geometricData = pd.read_csv(geometricDataFilename, engine="python", skipfooter=4)
        with open(cellTypeFilename, "w") as fh:
            fh.write("Label, Parent\n")
            for cellLabel in geometricData.iloc[:, 0]:
                if cellLabel in guardCells:
                    currentTypeId = stomataTypeId
                else:
                    currentTypeId = otherCellId
                fh.write(f"{cellLabel}, {currentTypeId}\n")


# def main():
#     parser = argparse.ArgumentParser(description="""With EasyMGXLabelExtractor.py you can extract the selected cell ids from MGX terminal by executing python3 EasyMGXLabelExtractor.py 'Picked label  608\\n
#     Picked label  606\\n
#     Picked label  485\\n' or by saving the text in a file  and using python3 EasyMGXLabelExtractor.py pathAndFilename""")
#     parser.add_argument("-textOrFilename", type=str, default="", help="Insert the filename or insert the text from MGX terminal and add parameter -t or --isText .")
#     parser.add_argument("-heat", "--heatMapFilename", type=str, default=None, help="Add the filename of the heat map to create a control heat map, which can be used to load the heat map.")
#     parser.add_argument("-t", "--isText", action="store_true", default=False, help="textOrFilename is interpreted as text.")
#     parser.add_argument("-save", "--filenameToSave", type=str, default=None, help="Specify the name to save the labels to.")
#     parser.add_argument("-ds", "--dontSave", action="store_true", default=True, help="Print the labels instead of saving it.")
#     parser.add_argument("-sel", "--saveSelection", action="store_true", default=True, help="Save also heat map with only selected cells.")
#     args = parser.parse_args()
#     initialDirectory = "./"
#     if args.textOrFilename == "" or args.heatMapFilename is None:
#         from tkinter import filedialog
#     if args.textOrFilename == "":
#         args.textOrFilename = filedialog.askopenfilename(initialdir=initialDirectory,
#                                                     title = "Select region defining file",
#                                                     filetypes = (("txt file","*.txt"), ("all files","*.*")))
#         assert not args.textOrFilename is "", "You need to select a file."
#     if args.heatMapFilename is None:
#         initialDirectory = Path(args.textOrFilename).parent
#         args.heatMapFilename = filedialog.askopenfilename(initialdir=initialDirectory,
#                                                     title = "Select full heat map file",
#                                                     filetypes = (("csv file","*.csv"),("all files","*.*")))
#         assert not args.heatMapFilename is "", "You need to select a file."
#     myConvertTextToLabels = ConvertTextToLabels(args.textOrFilename, args.filenameToSave,
#                                                 args.heatMapFilename, isText=args.isText,
#                                                 dontSave=args.dontSave,
#                                                 saveOnlyHeatMapSelection=args.saveSelection)
#     if args.filenameToSave is None and (args.dontSave or args.isText):
#         print(myConvertTextToLabels.GetLabels(onlyUnique=True))

if __name__ == '__main__':
    # main()
    mainCreateCellTypeOfProposedGuardCells()
