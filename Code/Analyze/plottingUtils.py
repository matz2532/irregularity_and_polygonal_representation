import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt

def createColorMapper(valueRange, cmapName="plasma", colorMap=None):
    norm = colors.Normalize(vmin=valueRange[0], vmax=valueRange[1], clip=True)
    if colorMap is None:
        colorMap = plt.get_cmap(cmapName)
    mapper = cm.ScalarMappable(norm=norm, cmap=colorMap)
    return mapper
