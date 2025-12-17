![shoreline_transect_intersection](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/e87b8d34-d9a4-4b1e-b3de-8e0be1c16ecd)

⚠️<b>Make sure you have downloaded the FES2014 or the FES2022 tide model following the guide 'How to Download the Tide Model'<b>⚠️

##  Tide Correction Button 

Ever since coastseg 1.1.0 released you can now use the `correct tides` button in coastseg to automatically correct the timeseries. This button will automatically find which files need to be tidally corrected and apply the tidal correction for you.

⚠️ <b>Important</b> ⚠️

If the tide model was NOT downloaded to `CoastSeg/tide_model` then the tide correction button will NOT work. The tide correction button will try to load the tide model from `<CoastSeg location>/tide_model` and an error will occur. If this happens move your tide model into `CoastSeg/tide_model` and run the notebook to clip the tide model.

### Steps

**WARNING**
You must have downloaded the tide model to the `tide_model` folder within coastseg for this button to work correctly.

1.**Load a Session**

- Load a session containing extracted shorelines.
- The following files will be saved to each ROI in the session (any existing files will be overwritten)

  -- `predicted_tides.csv`: contains the tide prediction from the tide model for each transect end point for each date a shoreline was detected on it.

  -- `tidally_corrected_transect_time_series_points.geojson` : contains the tidally_corrected shoreline transect intersections points from `raw_transect_time_series_merged.csv`.

  -- `tidally_corrected_raw_transect_time_series_vectors.geojson` : contains the tidally_corrected shoreline transect intersections from `raw_transect_time_series_merged.csv`. as vectors by connecting points that occurred on the same date together.

![select load session and tide correct](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/581f8b4a-062e-4326-9ae8-0145026fb9ad)

2.**Enter the beach slope**

- This is the beach slope in meters for the ROI you have selected.

3.**Enter the beach elevation**

- This is the beach elevation in meters relative to MSL(Mean Sea Level) for the ROI you have selected.

4.**Select the ROIs' ids to tidally correct**

  -- You should see some extracted shorelines on the map if you don't then the ROI ID won't appear in the dropdown

![select roi id for tide correction](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/9e212590-1f1e-4c51-b223-2e49a329a524)

5.**Click 'Correct Tides' and wait a few minutes.**

If you encounter any issues or have any questions please submit an issue at [CoastSeg Issues](https://github.com/Doodleverse/CoastSeg/issues/new/choose)

![load_session_correct_tides_demo](https://github.com/Doodleverse/CoastSeg/assets/61564689/d7a34d13-7c01-4a30-98b3-706a63195aa7)


## Credits

Thank you [DEA-Coastlines](https://github.com/GeoscienceAustrali/dea-coastlines/wiki/Setting-up-tidal-models-for-DEA-Coastlines) for making a guide on how to use pyTMD and [pyTMD](https://pytmd.readthedocs.io/en/latest/api_reference/aviso_fes_tides.html) for making a easy to use script to download the AVISO FES 2014 Model.
The `model_tides` in this code has been modified and the original function was originally written by Robbi Bishop-Taylor for the `dea-tools` package https://github.com/GeoscienceAustralia/dea-notebooks/blob/develop/Tools/dea_tools/coastal.py#L466-L473
