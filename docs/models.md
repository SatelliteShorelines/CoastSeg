# The CoastSeg Zoo Workflow

This workflow can be found in the 'SDS_zoo_classifier.ipynb' notebook.

The models available in this workflow come from [segmentation zoo](https://github.com/Doodleverse/segmentation_zoo) and were trained using [segmentation gym](https://github.com/Doodleverse/segmentation_gym).

## Installation Instructions

---

You'll need to follow the optional set of installation instructions to install the dependencies `tensorflow` and `transformers` into the `coastseg` environment in order to run the models.

**Warning**: The zoo workflow does not support Mac currently due to tensorflow and Mac having numerous compatibility issues. If you would like to help our team support Mac please submit an issue.

**Install Additional Dependencies**

- Only install these dependencies if you plan to use CoastSeg's Zoo workflow notebook.
- **Warning** installing tensorflow will not work correctly on Mac see for more details [Mac install guide](https://satelliteshorelines.github.io/CoastSeg/mac-install-guide/)

```bash
pip install tensorflow
pip install transformers
```

## Available Models

---

The following image segmentation models are available in CoastSeg to use on downloaded satellite imagery.

## RGB

**1.** `segformer_RGB_4class_8190958` : a segformer model that takes RGB imagery and applies a 4 class segmentation model

    Classes

    - 0: water
    - 1: whitewater,
    - 2: sediment,
    - 3: other

**2.** `sat_RGB_4class_6950472` : a resunet model that takes RGB imagery and applies a 4 class segmentation model

    Classes

    - 0: water
    - 1: whitewater,
    - 2: sediment,
    - 3: other

## NDWI

**1.** `segformer_NDWI_4class_8213427` : a segformer model that takes NDWI imagery and applies a 4 class segmentation model

    Classes

    - 0: water
    - 1: whitewater,
    - 2: sediment,
    - 3: other

## MNDWI

**1.**`segformer_MNDWI_4class_8213443`: a segformer model that takes MNDWI imagery and applies a 4 class segmentation model

    Classes

    - 0: water
    - 1: whitewater,
    - 2: sediment,
    - 3: other
