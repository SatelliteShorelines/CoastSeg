# Sign Up for Google Earth Engine

In order to use CoastSeg you must create a google cloud project with the Google Earth Engine API enabled.
This guide will walk you through how to create a cloud project and register it for the first time. You can read more [here](https://developers.google.com/earth-engine/cloud/earthengine_cloud_project_setup#get-access-to-earth-engine) about signing up for google earth engine.

### Step 1: Register

Visit: https://code.earthengine.google.com/register

![1_sign_up_screen](https://github.com/user-attachments/assets/17b4108a-32f3-4a54-b813-9c0a4bddb539)

### Step 2:
![2_paid_useage](https://github.com/user-attachments/assets/7415ff08-678f-4562-b313-2017d0da6593)


### Step 3: Create Project

1. Create a project and name it.
2. Scroll to the bottom and click the link to accept the terms

![3_create_project](https://github.com/user-attachments/assets/ae886d6a-7247-4080-8ac5-7536cb13901f)
![alt text](image-3.png)

### Step 4: Pop Up

When you click the link to accept the terms a pop up will open. Click agree and then click the button to continue

![pop_up](https://github.com/user-attachments/assets/0a864add-4d20-4f2d-a75b-2333297276af)

### Step 5: Confirm

1. Once you accept go back to the page you were on for step 3.
2. Confirm the project information

![4_confirm](https://github.com/user-attachments/assets/a8e9cbd9-93b1-404f-91c1-4da0c2bdda1c)

## Step 6: Finish Registering the Project


![success registered new project](https://github.com/user-attachments/assets/fe27985e-cc31-49fc-9187-1576ae8990f0)

After you click confirm the GEE console will open. Close it. We will not be using it.

![5_console](https://github.com/user-attachments/assets/33806a7b-c75f-4afa-a206-6d84ffa9399d)

## Step 7: Open the Google Cloud Console

Visit the [google console homepage](https://console.cloud.google.com/welcome)

Click your the project you registered in the console.

![select project in console](https://github.com/user-attachments/assets/97c8b47d-63ab-4c66-a188-2e98da2852ce)


## Step 8: Verify the Google Earth Engine API is Enabled 

Follw the guide the [Verify GEE API is Enabled](https://satelliteshorelines.github.io/CoastSeg/google-earth-enable-api/)

## Step 9: Open the Notebook

1. Open the notebook

```
conda activate coastseg
cd <location you installed CoastSeg>
jupyter lab SDS_coastsat_classifier.ipynb

```

2. Enter in your project ID 

My ID is 'ee-sf2309', but enter the id you entered with your email, then run the cell

```
initialize_gee(auth_mode = "notebook",project='ee-sf2309')
```

![gee id in notebook side by side](https://github.com/user-attachments/assets/3b291e2f-e772-4865-bd61-d5695295ff13)
