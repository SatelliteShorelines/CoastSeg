# How to Authenticate with Google Earth Engine(GEE)

## Why is it necessary to Authenticate with Google Earth Engine?

CoastSeg uses Google Earth Engine(GEE) to download satellite imagery. In order to use GEE an authentication token is needed. To get the GEE token you need to run the notebook cell containing the command ` ee.Initialize()` ,which either 1. loads in your GEE token if it exists or 2. prompts you to create a new token. GEE tokens last 1 week, so every week you will need to go through the process of authenticating with GEE with I have outlined below in the section **How to authenticate with Google Earth Engine(GEE) in a coastseg notebook**.

## How to get an Account with Google Earth Engine

You can access Google Earth Engine at https://signup.earthengine.google.com/. It takes about 1 day for Google to approve requests. You only need to do this step once. Once your request has been approved you can follow the following steps to authenticate with google earth engine with coastseg.

## How to authenticate with Google Earth Engine(GEE) in a coastseg notebook

![google earth engine auth tutorial_v2](https://user-images.githubusercontent.com/61564689/211117527-6af9d55f-d5a9-4d1a-b64b-d98d8e61a253.gif)

### If you're running coastseg locally:

1. Activate coastseg environment by running the following command on the Anaconda Prompt:

```bash
conda activate coastseg
```

2. Launch the notebook in your coastseg environment

```bash
jupyter lab <notebook name>
```

3.  Run the notebook cell containing the command `ee.Initialize()`.
    <br> The `earthengine authenticate` program will cause a web browser will open, log in with a Gmail account, and accept the terms and conditions. Then copy the authorization code into the indicated cell block into the notebook.

### If you're running coastseg in google colab:

1.  Run the notebook cell containing the command `ee.Initialize()`.
    <br> The `earthengine authenticate` program will cause a web browser will open, log in with a Gmail account, and accept the terms and conditions. Then copy the authorization code into the indicated cell block into the notebook.

### Errors with `gcloud`?

In the latest version of the earthengine-api, the authentication is done with gcloud. If an error is raised about gcloud missing, go to https://cloud.google.com/sdk/docs/install and install gcloud. After you have installed it, close the Anaconda Prompt and restart it, then activate the environment before running earthengine authenticate again.
