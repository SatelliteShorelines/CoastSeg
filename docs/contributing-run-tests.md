# How to test all the models in coastseg
This script will automatically test all the models in coastseg. Make sure the models in `available_models_dict` match the models currently available in coastseg. If they don't match the models currently available in coastseg update the `available_models_dict` dictionary as well as `parent_directory_names`.

Make sure to replace parameter after -P with path to ROI's RGB directory
`python test_models.py -P <your path here>"`

```
cd <location you installed coastseg>
cd debug_scripts
python test_models.py -P <your path here>"
```

### Example
`python test_models.py -P C:\development\doodleverse\coastseg\CoastSeg\data\ID_12_datetime06-05-23__04_16_45\jpg_files\preprocessed\RGB"`


