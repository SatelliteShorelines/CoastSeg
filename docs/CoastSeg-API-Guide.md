# CoastSeg Scripts Guide

This guide provides instructions on how to use the CoastSeg API to download imagery, extract shorelines, and apply tide correction.

## Contents

- [Script 1: Download Imagery and Initial Shoreline Extraction](#script-1-download-imagery-and-initial-shoreline-extraction)
- [Script 2: Extract Shorelines from a previously downloaded session](#script-2-extract-shorelines-from-a-previously-downloaded-session)
- [Applying Tide Correction (Optional)](#applying-tide-correction-optional)

## Prerequisites

- Ensure you have the CoastSeg conda environment set up and activated. If not, please follow the setup instructions provided in the CoastSeg documentation.
- Download the tide model required for tide correction. You can find instructions and download links [here](https://github.com/Doodleverse/CoastSeg/wiki/09.-How-to-Download-and-clip-Tide-Model).

## Running the Scripts

### Script 1: Download Imagery and Initial Shoreline Extraction

1.**Activate the CoastSeg Conda Environment:**

Open your terminal and activate the CoastSeg conda environment by running:

```bash
  conda activate coastseg
```

2.**Launch the Script**

- Navigate to the directory containing your script and run:

```bash
     python 1_download_imagery.py
```

The script performs the following actions:

- Initializes the Google Earth Engine.
- Downloads imagery based on specified regions of interest (ROIs).
- Extracts shorelines from the downloaded imagery.
- Optionally applies tide correction (uncomment the tide correction section if needed).

  3.**Understanding the Script Output:**

- The script will download imagery to the specified data folder.
- Extracted shorelines will be saved in the session's directory.
- Check the terminal for logs and messages indicating the progress and completion of tasks.

### Script 2: Extract Shorelines from a previously downloaded session

After completing the imagery download and initial extraction, you can proceed with the second script for further shoreline extraction.
Note: This script should be run after the first one because it loads the 'sample_session1' created by the first script

1.**Activate the CoastSeg Conda Environment:**

- Open your terminal and activate the CoastSeg conda environment by running:

```bash
  conda activate coastseg
```

2.**Uncomment Code to Apply Tide Correction (Optional)**

- The tide model MUST be downloaded as per the prerequisites. Follow a guide here on how to download it [How to Download Tide Model](https://satelliteshorelines.github.io/CoastSeg/How-to-Download-Tide-Model/)

- Uncomment the tide correction section in the script (shown below):
  - make sure to enter the `beach slope` and `reference_elevation`(relative to MSL) for your site.

```python
# coastsegmap.compute_tidal_corrections(roi_ids, beach_slope, reference_elevation)
```

3.**Launch the Script**

- Navigate to the directory containing your script and run:

- This script will use the ROI data downloaded by `1_download_imagery.py` and extract shorelines from the imagery.

- If you did not run `1_download_imagery.py` to download data then this script will not work

```bash
     python 2_extract_shorelines.py
```

It performs the following:

- Loads the download data from a session in 'CoastSeg/data' session created by the first script.
- Applies the new settings to control shoreline extraction from the downloaded imagery
- Optionally, applies tide correction (uncomment and configure if needed).
