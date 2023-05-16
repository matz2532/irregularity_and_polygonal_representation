def removeNotAllowedMeasure():
    import pickle
    # ktn1-2_20180618 ktn1-2 S1_0h ktn1-2/20180618 ktn1-2 S1/0h/ 23
    # ktn1-2_20180618 ktn1-2 S6_0h ktn1-2/20180618 ktn1-2 S6/0h/ 15, 29
    # ktn1-2_20180618 ktn1-2 S6_24h ktn1-2/20180618 ktn1-2 S6/24h/ 15
    # ktn1-2_20180618 ktn1-2 S6_48h ktn1-2/20180618 ktn1-2 S6/48h/ 19, 35
    # ktn1-2_20180618 ktn1-2 S6_72h ktn1-2/20180618 ktn1-2 S6/72h/ 22
    # ktn1-2_20180618 ktn1-2 S6_96h ktn1-2/20180618 ktn1-2 S6/96h/ 29
    filename = "Images/ktn1-2/20180618 ktn1-2 S6/96h/areaMeasuresPerCell.pkl"
    with open(filename, "rb") as fh:
        data = pickle.load(fh)
    data["labelledImageArea"].pop(29)
    # data["labelledImageArea"].pop(35)
    with open(filename, "wb") as fh:
        pickle.dump(data, fh)

def renameFilenameDictValues(printOutBeforeAndAfterValues=True, overwriteWithUpdate=False):
    import sys
    sys.path.insert(0, "./Code/DataStructures/")
    from MultiFolderContent import MultiFolderContent

    # select multiFolderContentsFilename and the key, which value will be changed by the selected function
    multiFolderContentsFilename = "Images/allFolderContents.pkl"
    valueOfKeyToChange = "cellContours"
    valueChangingFunction = lambda s: s.replace("cellContours.gpickle", "cellContour.pickle")

    # run or test just test how changes would look like
    multiFolderContent = MultiFolderContent(multiFolderContentsFilename)
    for folderContent in multiFolderContent:
        filenameDict = folderContent.GetFilenameDict()
        valueToChange = filenameDict[valueOfKeyToChange]
        changedValue = valueChangingFunction(valueToChange)
        if printOutBeforeAndAfterValues:
            tissueName = folderContent.GetTissueName()
            print(f"{tissueName=} {valueToChange=} {changedValue=}")
        filenameDict[valueOfKeyToChange] = changedValue
    if overwriteWithUpdate:
        multiFolderContent.UpdateFolderContents(printOut=True)

if __name__== "__main__":
    # renameFilenameDictValues()
    removeNotAllowedMeasure()
