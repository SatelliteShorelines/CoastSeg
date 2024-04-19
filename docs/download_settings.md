## Dates

- Select the start and end date to download imagery between
- You can change these and click download imagery to download more imagery for same ROI(s)

## Months

- To only download imagery for certain months hit the check boxes
- These months are saved in the 'months_list' to the right as the month number (eg. 1 for January etc.)

![download settings dates](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/d61a83a4-1d5e-4a23-81f0-118fb62831d9)

## Cloud Masking

- You can switch off cloud masking so that the images saved in 'Coastseg/data/<YOUR ROI>/jpg_files\preprocessed\RGB' don't have the cloud mask applied

- You can turn on/off cloud masks when extracting shoreline too

**Example of Masked Out Clouds**

![masked out clouds](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/461723b9-a4bc-41ec-a5b0-9ecc4eb318f7)

![apply cloud mask](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/74ee86b6-1fd4-40d6-b237-2bca4a7da074)

## Cloud Threshold

- This controls the maximum percentage of clouds allowed in the downloaded images

## Percentage Bad Pixels

- This controls the maximum percentage of bad pixels (cloud and no data combined) allowed in the downloaded images

![image](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/609f223f-c7e8-420a-9b3d-eb75421f0bbc)

## Resumable Downloads

To resume a partial download follow the following steps:

1.Click Load Session

2.Navigate to the 'CoastSeg/data' and load an ROI

3.Modify the date range, months list, or percentage of bad pixel settings

4.Click save settings

5.Click Download Imagery

![image download continue and skip](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/9a58d341-a2eb-482d-bc99-51714d53cf5c)

- In this screenshot you can see some images were skipped due to the cloud cover exceeded the maxmium percentage of clouds allowed set by the 'cloud thresh' settings

- In this screenshot you can see that for L8 21 images were available to download but 7 already had been downloaded so only 14 were remaining to download
