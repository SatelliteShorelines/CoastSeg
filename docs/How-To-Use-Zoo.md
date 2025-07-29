# How to Use the Zoo Notebook
- ⚠️ Ensure you have downloaded data from Google Earth Engine before running Zoo models
- ⚠️ Zoo notebook runs one region of interest (ROI) at a time

## Run Start Up Cell
![zoo_run_cell](https://github.com/user-attachments/assets/f67ccc77-7a59-401e-a0a7-ba15d4fba7cf)

## Step 1: Select a Model
- Select the Model input you would like to use (RGB, MNDWI, or NDWI)
- The NDWI and MNDWI will be automatically created using avalible RGB imagery

![zoo_step1](https://github.com/user-attachments/assets/803c1ab2-2687-4eaf-a6a7-8a876aeb0c49)

**Basic Model Settings**

- By default the global segformer model is selected

**Advanced Model Settings (Experienced Users Only)**

- Otsu Threshold (off by default)

![zoo_otsu](https://github.com/user-attachments/assets/da291475-124b-484c-a499-5b41a10df4ea)

- Time Test Augmentation (off by default)

![zoo_time_test](https://github.com/user-attachments/assets/93c2c307-b7c6-49f5-8ebe-24f470b5bdff)

## Step 2: Select Settings
- Select the settings you would like for your current session
- Make sure to click the save settings button to ensure your settings have been applied (Can check in the view settings tab)
- For more information on these settings see the [Extract Shorelines Settings Guide](https://satelliteshorelines.github.io/CoastSeg/extract_shoreline_settings/) 

![zoo_step2](https://github.com/user-attachments/assets/5990ba97-959d-4b36-a9bf-159d96a5fa63)


## Step 3: Upload Files
- Upload a GeoJSON file that contains either transects or shorelines
- If both the transects and shorelines are within the same ‘config_gdf.geojson’ file, you will need to upload the same file for the reference shoreline and the transects
- If no file is provided, CoastSeg will attempt to load an available file for you
	- If no transects or shorelines are available within the region of interest, an error will occur

![zoo_step3](https://github.com/user-attachments/assets/d7023052-9c0c-460e-974a-02d38869cc3e)

**Example of uploading transect GeoJSON file**

![zoo_step3_add_transects](https://github.com/user-attachments/assets/0c624ff2-5234-40e0-a7ee-767f55731d23)

## Step 4: Extract Shorelines with Model
1: Session Name
- Enter a name for your session. A new folder will be created with this name in the ‘sessions’ directory 

2: Select Images
- Select the RGB directory from your region of interest (roi) with downloaded imagery from the ‘data’ directory 

3: Run Model

![zoo_step4](https://github.com/user-attachments/assets/726ab090-9712-4430-89bd-dec528243212)

**Example of Step 4 set up**

Red arrow showing the RGB folder to be selected containing the images for shoreline extraction. Green arrow showing the final step to extract shorelines.

![zoo_step4_details](https://github.com/user-attachments/assets/e741a0a6-4e46-4ad7-bb19-a2dcff6b87ce)

**Example of a running session in progress**

- ⚠️ Zoo workflow takes longer on average to complete than the CoastSat workflow.
	- This is because the zoo workflow loads a more advanced model that takes longer to process all the imagery in a session.
	- In the screeshot below you can see the progress bar, called "Applying Model", updates as the zoo model finishes running on each image. (note the warnings are normal)

![zoo_step4_run_model](https://github.com/user-attachments/assets/3776be97-5e2b-42c4-a113-9d7245944826)
![zoo_step4_run_model_pt2](https://github.com/user-attachments/assets/93138876-085e-4511-82b9-0bfa0a7d9df2)

## Step 5: Tidal Correction 
- Ensure the tide model has been downloaded to CoastSeg/tide_model for tidal correction to work 
- Follow the guide to download the [tide model](https://github.com/SatelliteShorelines/CoastSeg/wiki/09.-How-to-Download-Tide-Model)
- Ensure shorelines have been extracted prior to tidal correction. Not all imagery will contain suitable shorelines and result in tide correction being not possible

Step 1: Section a Session
- Select a session from the ‘sessions’ directory that contains extracted shorelines

Step 2: Run Tidal Correction
- Runs the tide model and saves tidally corrected CSV files in the selected sessions directory 

- For appropriate [slope formats](https://satelliteshorelines.github.io/CoastSeg/slope-file-format/)

- For appropriate [tide formats](https://satelliteshorelines.github.io/CoastSeg/tide-file-format/)

Select between the FES 2014 or FES 2022 model before clicking correct tides.

- Tidal correction will take a few minutes to run because the tide model is large, several GB

![zoo_step5](https://github.com/user-attachments/assets/2b88114d-ea56-4f1d-a7c2-edb00fc38d7e)
