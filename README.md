## Microtubule-mediated cell shape regulation contributes to efficient cell packing in Arabidopsis thaliana cotyledons
Timon W. Matz<sup>1,2</sup>, Ryan C. Eng<sup>3</sup>, Arun Sampathkumar<sup>4</sup>, Zoran Nikoloski<sup>1,2</sup>

<sup>**1**</sup> Bioinformatics, Institute of Biochemistry and Biology, University of Potsdam,14476 Potsdam, Germany;

<sup>**2**</sup> Systems Biology and Mathematical Modelling, Max Planck Institute of Molecular Plant Physiology, 14476 Potsdam, Germany;

<sup>**3**</sup> Department of Comparative Development and Genetics, Max Planck Institute For Plant Breeding Research, 50829 Cologne, Germany

<sup>**4**</sup> Plant Cell Biology and Microscopy, Max Planck Institute of Molecular Plant Physiology, 14476 Potsdam, Germany
# Abstract
Recent advances have started to uncover the mechanisms involved in organ and cell shape regulation. However, organizational principles of epidermal cells in different tissues remain poorly understood. Here, we show that polygonal representations of cotyledon pavement cells (PCs) in Arabidopsis thaliana exhibit increasing irregularity in side lengths and internal vertex angles during early stages of development. While the shape of PCs in cotyledons is more complex than that of cells in the shoot apical meristem (SAM), the polygonal representations of these cells share similar irregularity of side length. Comparison of the surface cell area with the area of the regular polygons, having optimally spaced tri-cellular junctions, reveals suboptimal junction placement for coverage in cotyledons and SAM. We also found that cotyledons show increased packing density compared to the SAM, indicating that PCs forgo coverage of larger areas to potentially increase tissue stability. The identified shape irregularity and cell packing is associated with microtubule cytoskeleton. Our study provides a framework to analyze reasons and consequences of irregularity of polygonal shapes for biological as well as artificial shapes in larger organizational context.

[Link to pre-print of paper on BioRxiv]( https://doi.org/10.1101/2023.05.16.540958 )

# How to use the code
__System requirements:__

The code is written in python 3.8.1 using the following packages: matplotlib==3.6.2, networkx==2.8.4, numpy==1.23.5, pandas==1.4.4, scipy==1.10.0, seaborn==0.11.2, shapely==2.0.1, sklearn==1.2.0

Either use anaconda/miniconda or virtualenv to create a virtual environment with the specific python and package versions. Alternatively, you can use the 'environment.yml' file to create the needed environment with the command ´conda env create -f environment.yml´ and activate the environment with ´conda activate EnvironmentName´ before executing any script. 
If you have not installed python, you can also install python with the packages versions directly on your computer.

__How to run the code (top-down perspective):__

Execute the code from the base folder (so that you are in the folder of the README.md file).
1. Run __\_\_mainCreateMeasures\_\_.py__ to create the irregularities and polygonal area ratios of the pavement cells from WT full cotyledons of 120h post germination (original data), pavement cells tracked over time from Eng et al. 2021, and cells from the central region of the shoot apical meristem from Matz et al. 2022. The measures are combined in the 'combinedMeasures_tissueTypeName.csv' with the tissueTypeName referring to 'full cotyledon', 'Eng2021Cotyledons', or 'Matz2022SAM', respectively.
2. Run __\_\_mainVisualizeResults\_\_.py__ to create visualizations of results from pre-print paper.


To properly create the measure data, you actually need the outlines and polygonal representations as .ply-files. However, due to size limitations I could not yet upload these files. Please, contact me to arrange data transferal. I'm sorry for the inconvenience. 
 

__To run the code with your own data:__

For easier integration prepare your data in the following folder structure create a sub-folder of your tissue type, and nest your genotypes and replicates as separate folders inside with each respective time point of the replicate getting another sub-folder in which you add the respective data, see example:

- Images
  - Eng2021Cotyledons
    - col-0
      - 20170327 WT S1
        - 0h
        - ...
        - 96h

There are 2 ways to create the irregularity and polygonal area ratios from raw data. A) You can either get the neccessary files from using MGX or B) extract the labelled image and contours of cells from for example GraVis.


__For A) you need to:__
1. Create the segmented surface of your tissue
2. Extract the geometric data, e.g. cell position and size, of the tissue using 'Process/Mesh/Heat Map/ Heat Map Classic' and saving it under the respective folder with the suffix '_geometricData.csv' or '_geometricData full.csv', depending on whether you are using A1) mainOnSAMMatz2022() or A2) mainCalculateOnNewCotyledons().

&ensp;&ensp;&ensp;&ensp;Choose A1) when you don't want to exclude any cells, but A2) when you want to exclude for example stomata or other cells. In case of A2), you either need to select (left-clicking + CTRL + SHIFT) the cells to ignore with the pipette tool in MGX and copy (!!! Do not CTRL + C, but rather right click and select copy, to avoid exiting MGX !!!) the text from the terminal (looking like 'Picked label 12345') saving the output in a new text file with the suffix '_proposed guard cells.txt'. An alternative for A2) is to save the surface under a new name and delete all unwanted cells (using the fill tool having no cell label selected left-clicking with CTRL + SHIFT + ALT selected) and saving the geometric data with the suffix '_geometricData no stomata.csv'.
    
3. Extract data used for contour extraction on the original segmented surface converting the surface into a cell mesh using 'Process/Mesh/Cell Mesh/Convert to a cell mesh' with the 'Max Wall parameter' set to -1 and saving it as a ply.-file with 'Process/Mesh/Export/Cell Graph Ply File Save' with the suffix '_full outlines.ply' or '_halved outlines.ply' for A1) and A2), respectively. 
4. Extract tri-way junction positions saving the original segmented surface under a different name and converting the mesh into a cell mesh with the 'Max Wall parameter' set to 0 and saving this surface as a ply.-file with the suffix '_only junctions.ply'. Also make sure you have the cell graph of the respective file (suffix should be '_only junctions_cellGraph.ply') copied in the tissue's folder.
5. Extract the geometric data of the polygonized surface and save with the suffix '_geometricData poly.csv'.
6. Run __\_\_mainCreateMeasures\_\_.py__ after changing scenario name to your respective folder name and commenting the unused functions at the bottom out (adding #-symbol infront of all of the three functions except mainOnSAMMatz2022() or mainCalculateOnNewCotyledons() )

__For B) you need to:__
1. Provide the original image, the labelled image (having the same size as the original image, with the pixel values corresponding to whether it is a background: 0, cell contour: 1, or a cell label: values >= 2), and the contour information of the cells (for now this is just implemented for a special text file format, but can be easily changed on request for other formats). The cells, for which contours are provided, will be selected for measure creation. Both files are easiest created by using [GraVis stand-alone executable](https://github.com/jnowak90/GraVisGUI/releases), where you select your tissue and let GraVis do the segmentation of your image. For more details see the [original paper](https://doi.org/10.1038/s41467-020-20730-y) and the [code on GitHub](https://github.com/jnowak90/GraVisGUI/).
2. Adapt the InputData.py file similar to the original file adding your genotypes and replicate names adapting the file names. The position of the list entry corresponds to the name given under timePoints, specifiying the time point folder name.
3. Run __\_\_mainCreateMeasures\_\_.py__ after changing scenario name to your respective folder name and commenting the unused functions at the bottom out (adding #-symbol infront of all  of the three functions except mainCalculateOnEng2021Cotyledon() ). During runtime a window will open check tri-way junction positioning close the window, when you are satisfied with the junction positions. Left-click close to any missing junction and the point closest, where two cells meet will be selected as a junction (when you click a second time at the exact same position you force this pixel to be a tri-way junction and can delete the other undesired one by right-clicking close to it). 

(You can also change the all default suffixes as you desire.)


For further questions, do not hesitate to write me or open a request.
