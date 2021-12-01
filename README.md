# CoastSeg
The mapping extension for Zoo; carry out image segmentation on geospatial datasets

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
