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

1. **Activate the CoastSeg Conda Environment:**
   Open your terminal and activate the CoastSeg conda environment by running:
   ```bash
     conda activate coastseg
   ```
2. **Launch the Script**

- Navigate to the directory containing your script and run:

```bash
     python 1_download_imagery.py
```

The script performs the following actions:

- Initializes the Google Earth Engine.
- Downloads imagery based on specified regions of interest (ROIs).
- Extracts shorelines from the downloaded imagery.
- Optionally applies tide correction (uncomment the tide correction section if needed).

3. **Understanding the Script Output:**

- The script will download imagery to the specified data folder.
- Extracted shorelines will be saved in the session's directory.
- Check the terminal for logs and messages indicating the progress and completion of tasks.

### Script 2: Extract Shorelines from a previously downloaded session

After completing the imagery download and initial extraction, you can proceed with the second script for further shoreline extraction.
Note: This script should be run after the first one because it loads the 'sample_session1' created by the first script

1. **Activate the CoastSeg Conda Environment:**
   Open your terminal and activate the CoastSeg conda environment by running:
   ```bash
     conda activate coastseg
   ```
2. **Apply Tide Correction (Optional)**

- Ensure you have downloaded the tide model as per the prerequisites.
- Uncomment the tide correction section in the script:
  - make sure to enter the `beach slope` and `reference_elevation` for your site.

```python
# coastsegmap.compute_tidal_corrections(roi_ids, beach_slope, reference_elevation)
```

3. **Launch the Script**

- Navigate to the directory containing your script and run:

```bash
     python 2_extract_shorelines.py
```

It performs the following:

- Loads the session created by the first script.
- Applies new settings for shoreline extraction.
- Optionally, applies tide correction (uncomment and configure if needed).

## Applying Tide Correction (Optional)

To apply tide correction:

1. **Ensure you have downloaded the tide model as per the prerequisites.**

2. **Uncomment the tide correction section in each script:**

```python
# coastsegmap.compute_tidal_corrections(roi_ids, beach_slope, reference_elevation)
```

Replace beach_slope and reference_elevation with the appropriate values for your region of interest. 3. **Re-run the script after uncommenting and modifying the tide correction section.**

Remember, tide correction is optional and should be applied based on your specific requirements and the characteristics of the imagery and region of interest.
