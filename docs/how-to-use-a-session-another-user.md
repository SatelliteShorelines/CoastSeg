If you have a session from another user you want to load into CoastSeg follow these steps

## Guide 1 : You have the downloaded ROI imagery and the extracted shoreline session

**1.Move the ROI folders containing the imagery into CoastSeg/data**

**2.Move the new session folder(s) into CoastSeg/sessions**

**3.Click 'Load Session' and load the session folder from CoastSeg/sessions**

## Guide 2: You DO NOT have the downloaded ROI Imagery but DO have the extracted shoreline session

**1.Move the new session folder(s) into CoastSeg/sessions**

**2.Click 'Load Session' and load the session folder from CoastSeg/sessions**

**3.Get the Warning Message**

- A warning called 'Warning Missing Data' will appear listing the missing ROI IDs. In this case is ROI 'fvk3'

- Hover over the ROIs on the map (in red) and check the ROI drop download on the upper right to see the ID (hovering over ROI 'fvk3' in the screenshot )

**4.Download the ALL the ROIs**

- Click the ALL the ROIs and click 'Download Imagery' to download the data for these missing ROIs

  -- Don't worry the code will automatically detect the ROI you already downloaded and won't download it, but you must select ALL the ROIs. This is because under the hood this creates a new session containing only the ROIs you have selected.

  -- The download settings for the session will already have been loaded into so don't worry about changing them

![warning missing rois loaded from session](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/2c67676e-cf86-4493-884d-452bda356d26)

- As you can see in the screenshot below the first ROI downloads all the missing imagery, while the second ROI which we had the downloaded data for only downloads a single image because 5 images already exist.

![partial download 2 rois](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/86de7b4a-51de-47b8-9a00-b9d903aded39)
