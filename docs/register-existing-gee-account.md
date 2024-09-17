# Register existing account with Google Earth Engine

This tutorial is for existing users of CoastSeg who need to use a Google Cloud Project in order to continue using GEE.

### Step 1: Check if you have a cloud project


1. Visit: [Google Earth Engine Registration](https://code.earthengine.google.com/register)
2. Sign in with the email you have used with CoastSeg


![1_sign_up_screen](https://github.com/user-attachments/assets/17b4108a-32f3-4a54-b813-9c0a4bddb539)

### Step 2:
![2_paid_useage](https://github.com/user-attachments/assets/7415ff08-678f-4562-b313-2017d0da6593)


### Step 3: Choose an existing project

If you have no created a project before then follow the [register for a new cloud project guide](https://satelliteshorelines.github.io/CoastSeg/google-earth-engine-setup/)

![choose existing project to register](https://github.com/user-attachments/assets/5073137c-9b54-4410-b397-be4604e0d6a8)

### Step 4: Register the Project

1. If your project is already registered it will state "project already registered"

![project already registered](https://github.com/user-attachments/assets/c785fb7d-50d2-41d1-a903-638856b6870d)

2. If the project is NOT registered it will ask you to confirm your registration

![register existing project confirm screen](https://github.com/user-attachments/assets/dd468b47-4d46-4201-9c4c-c0c0c471a1cc)


### Step 5: Finish Registering the Project

![success registered new project](https://github.com/user-attachments/assets/fe27985e-cc31-49fc-9187-1576ae8990f0)

After you click confirm the GEE console will open. Close it. We will not be using it.

![5_console](https://github.com/user-attachments/assets/33806a7b-c75f-4afa-a206-6d84ffa9399d)

## Step 6: Open the Google Cloud Console

Visit the [google console homepage](https://console.cloud.google.com/welcome)

Click your the project you registered in the console.

![select project in console](https://github.com/user-attachments/assets/97c8b47d-63ab-4c66-a188-2e98da2852ce)

## Step 7: Verify the Google Earth Engine API is Enabled 

Follw the guide the [Verify GEE API is Enabled](https://satelliteshorelines.github.io/CoastSeg/google-earth-enable-api/)

## Step 8: Open the Notebook

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