# Overview

This guide will walk you through signing up for the necessary account and then choosing between two methods to download and clip the tide model for use in CoastSeg.

# Part 1: Register for the Tide Model
You only need to do this part once. Be sure to save your username and password in a safe place. 
Be aware that AVISO may update their registration guide at any time and this page may be out of date.

### Step 1: Register
- **Sign Up:** Begin by registering on the AVISO platform. Visit the [AVISO Registration Page ](https://www.aviso.altimetry.fr/en/data/data-access/registration-form.html) to create your account.

### Step 2: Select the Tide Model

- **Select the FES Model** During the registration process, you'll be presented with various data models to choose from. Ensure you select the **FES (Finite Element Solution - Oceanic Tides Heights)** model.


### Step 3: Account Verification
- **Confirmation Email:** After completing the registration, AVISO will send you a confirmation email. Check your inbox (and spam folder if necessary) for this email.
- **Verify Your Account**: Click on the link provided in the email to verify your account. This step is crucial to activate your account and gain access to the data models.

### Step 4: Accessing Your Dashboard
- **Login:** Once your account is verified, [log into your AVISO account.](https://www.aviso.altimetry.fr/en/my-aviso-plus.html)
- **Navigate to Products:** On your account dashboard, you'll find a left side menu. Click on the **My products** option.
![image](https://github.com/Doodleverse/CoastSeg/assets/61564689/bf5382f0-6bc0-4867-893d-c8f84a3d3760)

### Step 5: Confirm Your Subscription
- **Check Your Subscriptions:** In the "Your current subscriptions" section, ensure that the **FES (Finite Element Solution - Oceanic Tides Heights)** model is listed. This confirms that you've successfully subscribed to the desired model.

![FES_products](https://github.com/user-attachments/assets/87d23089-1f46-43c7-bb96-d1850dc7a9c4)

# Part 2: Download and Clip the Tide Model

### Space and Time Requirements

- **Storage Space:** Ensure you have at least 14GB of free space available for fes2014.
   - If you want both fes2014 and fes2022 you will need at least 35 GB of free space.
- **Time Commitment:** The download and setup process for the fes2014 model takes approximately 1-2 hours. This is a one-time requirement.

## Download_tide_model.ipynb Notebook

The easiest way to download and clip the tide model is to use this notebook. Run all the code in the notebook and you're done!

1. **Prepare:** Ensure you have you are in an activated coastseg environment and know your AVISO email and password.
2. **Open Notebook:** Launch `Download_tide_model.ipynb` in Jupyter Notebook.

```
cd CoastSeg
jupyter lab Download_tide_model.ipynb
```

3.**Run Code:** Execute all cells in the notebook. The notebook will guide you through the download and clipping process automatically.

4.**Check Results:** Confirm that the model files are correctly downloaded and clipped in the specified directory.

- **Example Output from Downloading Tide Model Step in Notebook**
  ![download_tide_model_notebook](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/eecae8d2-cb5e-4e50-a587-3260ff9469b7)

- **Example Output from Clipping Tide Model Step in Notebook**
  ![clip_tide_model](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/abba33ae-1c63-4c48-be14-4e51e8224870)

5.**Downloaded Tide Model Contents**

- Below is an example of the CoastSeg tide model contents for FES 2014 after the model was downloaded and clipped.

![tide model contents](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/2a8f2425-993f-4184-90cf-01b6ff65af4a)

### Why is the tide model clipped?

To make the tide predictions compute faster we will clip the model it to various regions across the planet. After we've clipped the model to each region, when we want to predict the tide at a particular latitude and longitude we first figure out which region the point lies within, then we load the tide model we clipped to that region.

### Structure of tide model in CoastSeg

```
├── CoastSeg
|
|___tide_model
|    |_ fes2014
|    |     |_load_tide
|    |     |       |__2n2.nc.gz
|    |     |       |__eps2.nc.gz
|    |     |       |__ ....
|    |     |_ocean_tide
|    |     |       |__2n2.nc.gz
|    |     |       |__eps2.nc.gz
|    |     |       |__ ....
|    |
|    |_ region0
|    |     |_fes2014
|    |     |       |__load_tide
|    |     |       |       |__2n2.nc
|    |     |       |       |__eps2.nc
|    |     |       |       |__....
|    |     |       |__ocean_tide
|    |     |       |       |__2n2.nc
|    |     |       |       |__eps2.nc
|    |     |       |       |__....
|    |_ region1
|    |_ region2
|    |_ region3
|    |_ region4
|    |_ region5
|    |_ region6
|    |_ region7
|    |_ region8
|    |_ region9
|    |_ region10

```

## Running into Problems?

---

1. Your account may not be verified yet, which means you can't download the model
2. Occasionally the AVISO server goes down. Wait a few hours and try again and see if its working.

### Troubleshooting

If you are running coastseg on a secure network you may need to make the following modifications to your `.condarc` file.

1. If you get an error message similar to `CondaValueError: You have chosen a non-default solver backend(libmamba) but it was not recongized`
   - Solution: comment out the line in the .condarc file `solve:libmamba`
2. You may also need to modify the `ssl_verify` to either True or False depending on your network security.
   - Always switch `ssl_verify` to True if you ever set it it to False for debugging purposes.

# How to Perform Tide Correction

⚠️ You must have downloaded the FES2014 and/or FES2022 tide model before attempting to correct tides

## ⚠️ Important: Read Before Using Tide Correction Button

If the tide model was NOT downloaded to `CoastSeg/tide_model` then the tide correction button will NOT work. The tide correction button will try to load the tide model from `<CoastSeg location>/tide_model` and an error will occur. If you downloaded the tide model to a different location please move it to `CoastSeg/tide_model` and be sure to clip the tide model.

For detailed instructions on performing tide correction, see [How to Tidally Correct](How-to-Tidally-Correct.md).


## Credits

Thank you [DEA-Coastlines](https://github.com/GeoscienceAustrali/dea-coastlines/wiki/Setting-up-tidal-models-for-DEA-Coastlines) for making a guide on how to use pyTMD and [pyTMD](https://pytmd.readthedocs.io/en/latest/api_reference/aviso_fes_tides.html) for making a easy to use script to download the AVISO FES 2014 and FES 2022 Model.
The `model_tides` in this code has been modified and the original function was originally written by Robbi Bishop-Taylor for the `dea-tools` package https://github.com/GeoscienceAustralia/dea-notebooks/blob/develop/Tools/dea_tools/coastal.py#L466-L473 . For more informaion on the FES 2014 model please visit https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/global-tide-fes/description-fes2014.html and https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/global-tide-fes/release-fes22.html for the FES 2022 model. 
