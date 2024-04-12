# Filtering Imagery
Extracting the best shorelines requires that only the best imagery is used which means that bad imagery needs to be sorted out. You should filter out bad imagery in `data/roi_id/jpg_files/preprocessed/RGB` by moving any bad imagery to a designated subdirectory named 'bad'.

It is important to note that this operation does not delete any TIFF files, but rather helps to improve the efficiency of shoreline extraction and enhance the quality of the extracted shorelines. Check out the [wiki page about the shoreline extraction process](https://github.com/Doodleverse/CoastSeg/wiki/6.-How-to-Extract-Shorelines) for more information about how to extract shorelines.

## Step by Step Guide
### Before you begin
Download data with the `SDS_coastsat_classifier.ipynb` and make sure it downloaded to your `data` directory within coastseg
1. Open the `data` directory within coastseg
2. Navigate to the ROI directory
- If you're not sure which ROI directory you used for a particular session go to the `sessions` directory, click the session you are interested in and open the `config.json` file. Look for the variable `sitename` and the sitename will be in a format similar to `ID_yvk1_datetime06-05-23__07_07_42`. This is the name of the ROI directory containing the downloaded data in the `data` directory within coastseg

### Example config file from data
![sitename, smooth   tasty  (500 × 600 px) (500 × 530 px)](https://github.com/Doodleverse/CoastSeg/assets/61564689/cfa07a84-9768-4659-8fb4-794f53ff82cd)
### Example config file from sessions
![sitename, smooth   tasty  (500 × 600 px) (500 × 530 px) (500 × 250 px)](https://github.com/Doodleverse/CoastSeg/assets/61564689/f32c97cf-84c4-4943-be79-18f5662689b0)

3. In the ROI directory navigate to `jpg_files>preprocessed>RGB`
- Example on windows: `CoastSeg\data\ID_yvk1_datetime06-05-23__06_57_26\jpg_files\preprocessed\RGB`
4. Remove the files the you don't like
    1. Create a new subdirectory named 'bad' within the data/roi_id/jpg_files/preprocessed/RGB directory if it does not already exist.
    2. Identify the files within the data/roi_id/jpg_files/preprocessed/RGB directory that you want to remove.
    3. Copy the files you wish to remove and paste them into the 'bad' subdirectory created in step 1. - Confirm that the files are successfully copied to the 'bad' subdirectory.

![coastseg_screenshot_bad_subdir](https://github.com/Doodleverse/CoastSeg/assets/61564689/8ac4d26b-2ec1-4d1d-a4f2-8d7857ac11b8)



5. Test the shoreline extraction process to verify that the removal of files from the `data/roi_id/jpg_files/preprocessed/RGB` directory has improved the efficiency and quality of the extracted shorelines. Check out the [wiki page about the shoreline extraction process](https://github.com/Doodleverse/CoastSeg/wiki/6.-How-to-Extract-Shorelines) for more information about how to extract shorelines.


 