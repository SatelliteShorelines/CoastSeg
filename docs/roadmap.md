## :earth_americas: CoastSeg RoadMap

### Version 3

Will provide 'additional functionality', which can be categorized as follows:

a) More performant shoreline extraction: the goal is fast and accurate shoreline extraction, with known errors. Key goals include 1) outlier detection, 2) faster shoreline extraction, and 3) improved shoreline detection in more diverse settings, e.g. ice, marsh and other vegetation, rocky cliffs, gravel beaches, etc

b) Improved imagery: the goal is to use ML to enhance imagery, where possible. Key goals include 1) image super-resolution, 2) image inpainting, and 3) automated image registration

c) Additional image sources. The initial focus will be on integrating with Planet Labs' Planetscope (3-m) imagery

A major goal is to incorporate more data-driven approaches to shoreline detection using Machine Learning sub-components. We track progress using the [project board](https://github.com/orgs/Doodleverse/projects/5/views/2) and in [issues](https://github.com/Doodleverse/CoastSeg/issues). If you have an idea for a new feature not listed here, please get in touch!

#### Major features:

- Faster Shoreline Extraction
- Faster Image Segmentation
- Faster Batch Image Downloading
- New Models
  - 4 class B+NIR+SWIR Segformer
  - 4 class NDWI Segformer
  - 4 class MNDWI Segformer
- Image Filtering
  - Automated : metric comparison with base image
  - Automated : Machine Learning model Sniffer 🐕 designed to sort bad imagery out

* CoastSeg allows users to filter out images by placing images into a bad directory to not use them for shoreline extraction

- More Shoreline Filter for Shoreline Extraction
  - vertex simplification
  - outliers based on time-average shoreline
  - discontinuous shorelines
- Image Sources
  - Planet 3m
- Super Resolution
- Image Registration
- Image Inpainting - Stable Diffusion
- Docker container
- New Frontend?
  - Goal to move away from jupyter potentially for better performance and hosting
