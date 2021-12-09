# CoastSeg

(in the planning stages only)

A mapping extension for CoastSat and Zoo; carry out image segmentation on coastal satellite imagery datacubes

In a nutshell ...

![slide1](https://user-images.githubusercontent.com/3596509/144157212-cd37e06f-29d4-4cb9-b91d-da2568e45a39.PNG)

![slide2](https://user-images.githubusercontent.com/3596509/144157226-d3d0d252-27be-47d7-a1c7-8220812c0652.PNG)

![slide3](https://user-images.githubusercontent.com/3596509/144157239-61660733-5c94-40bb-961d-010ea6d26ce7.PNG)

![slide4](https://user-images.githubusercontent.com/3596509/144157265-c1a05e7d-98a4-4f20-8e0d-9664e6e0ad9b.PNG)

![slide5](https://user-images.githubusercontent.com/3596509/144157283-6d28eaf6-ab5a-46c6-9658-a38138ed66f9.PNG)

![slide6](https://user-images.githubusercontent.com/3596509/144157298-b66d352b-28a4-41f0-8bb0-898294114c28.PNG)

![slide7](https://user-images.githubusercontent.com/3596509/144157319-2df04a8f-7dc3-4ce9-84ed-5957bcdc2b37.PNG)

![slide8](https://user-images.githubusercontent.com/3596509/144157327-33b50bfc-6f72-4ec7-9a01-ef86e4dc2bf6.PNG)

This tool will facilitate Doodler and Zoo style image segmentation on geospatial imagery, by combining the following elements

### A local ipyleaflet tile server for data 
- [ ] [Local Tile Server for Geospatial Rasters](https://github.com/banesullivan/localtileserver?s=09#ipyleaflet-tile-layers)
- [ ] all ipyleaflet and folium functionality for data exploration and retrieval

### Cloud-based image retrieval
- [ ] coastsat for image retrieval
- [ ] also [wxee](https://github.com/aazuspan/wxee) for image retrieval and xarray support

### Parallelism and modern geospatial formats
- [ ] xarrays facilitate segmentation workflows in Dask
- [ ] COGs

### Doodler for geospatial imagery
- [ ] interface imagery for doodler
- [ ] wrapper utilities to convert labels to geospatial formats

### Zoo for geospatial imagery
- [ ] interface for preparing datasets for Zoo model training and evaluation
- [ ] interface for applying trained models to geospatial imagery

### Basic workflow

1. User launches an interactive map server (localtileserver)
2. User defines AOI, and other relevant spatial operations
3. CoastSat functionality downloads imagery, archives, cloud masking, tidal corrections
4. Data converted to xarrays
5. Zoo models act on xarray datacubes

### This toolbox will attempt to be fully automated

this toolbox will attempt to use semantic segmentation on deep learning models trained on large labeled datasets 

this will hopefully cirumvent two limitations in the current CoastSat workflow
- Coastsat uses a user-defined reference shoreline
- Coastsat relies on user-modified classifiers

The expected user experience will be:

1. define ROI graphically / interactively
2. use coastsat to download imagery
3. user selects a pre-trained segmentation model
4. user applies model to a large region
5. optionally, user interactively refines segmentation by selecting different or more models to ensemble until they are satisfied with the result
5. outputs would replace the segmentation outputs of coastsat (or they could be cleverly combined, TBD)
6. we would modify CoastSat Shoreline change analysis functions in a web map (ipyleaflet) environment
7. we would apply CoastSat tidal corretions
8. Install and activate the FES global tide model (like Inlet Tracker)
