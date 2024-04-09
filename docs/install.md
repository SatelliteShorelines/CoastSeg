## Regular users on Windows and Linux

Please refer to the [installation guide](https://github.com/Doodleverse/CoastSeg#installation-instructions) on the README

## Mac users

CoastSeg requires Tensorflow (TF), which doesn't play nicely with Mac. We advise you to use either Linux or Windows, if you can. We cannot troubleshoot Mac installations, but we can offer the following advice:

- TF for mac has its own instructions: https://developer.apple.com/metal/tensorflow-plugin/
- New Mac silicon runs TF, (and has its own TF branch), but the old intel Mac chips might not work with parts of TF.
- We are not sure if TF is compatible with M2 macs
- Our continuous integration tests check only the 'latest' version of Mac OS.
- If you get a working installation on Mac, please let us know, and we can edit these pages to communicate better advice. Thanks in advance

## Users working over secure network

First, some notes:

- ⚠️ The CoastSeg team is aware of issues with installing CoastSeg's conda dependencies ⚠️
- ⚠️The following instructions are valid as of 2023-09-12. If this proposed solution fails for you, please create a new [Issue](https://github.com/Doodleverse/CoastSeg/issues) ⚠️)
- ⚠️ Where possible, we advise you install the initial Coastseg conda environment not over a secure network. If and _ONLY_ if that is not possible, please follow the instructions posted below carefully
- If you are running these commands on a secure network verify you are connected to your VPN before running the following commands
- if you get an SSL error while running any of the `conda install` commands try the following command
- ONLY RUN THE PIP INSTALL COMMANDS AFTER CREATING A PIP.INI file [Go to Phase 1: Create a pip.ini File](#phase-1-create-a-pipini-file)

### Quick Start

Second, you must temporarily (for a few minutes) disable SSL verification. This should be done only once, when setting up the initial conda environment. If you find yourself having to do this for other instructions listed below and elsewhere, something is wrong and you should either start again from scratch, or raise an Issue if there is a legitimate bug. Coastseg developers are not liable for any unintended effects of ignoring the advice posted here, i.e. leaving `ssl_verify` to be `False` outside of the specific and unique case explained here.

```
conda config --set ssl_verify False
```

Now, you can install the rest of the dependencies. You must do this immediately.

```
conda install -c conda-forge geopandas jupyterlab -y
pip install coastseg
pip uninstall h5py -y
conda install -c conda-forge h5py -y
```

Finally, re-enable SSL verification. You _must_ do this _immediately_ after installing the conda packages above. If any of the above commands fail for whatever reason, you must also run this command to re-enable SSL. Coastseg developers are not liable for any unintended effects of ignoring the advice posted here, i.e. leaving `ssl_verify` to be `False` outside of the specific and unique case explained here.

```
conda config --set ssl_verify True
```

### Detailed Guide

#### Phase 0: Allow Git to work over a secure network

1. Configure git to use ssl

- Make sure to replace `"C:/Users/ulastname/Documents/DOIRootCA2.cer"` with your user name

```
git config --global http.sslcainfo "C:/Users/ulastname/Documents/DOIRootCA2.cer"
```

2. Test if this worked

- This step assumes you have a github account and are in the CoastSeg directory

```
git pull origin main
```

#### Phase 1: Allow Pip to work over a secure network

1. Open the `pip.ini` file
   - In anaconda prompt or the command line run:
   ```
   notepad %APPDATA%\pip\pip.ini
   ```
2. Edit the `pip.ini` file
   - Make sure to replace C:\Users\ulastname\Documents\cacert_with_doi.pem`
   - save and exit after editing
   ```
    [global]
    cert = C:\Users\ulastname\Documents\cacert_with_doi.pem
   ```
3. Check the contents of the pip.ini
   ```
    pip config list
   ```
   - it should return something like
   ```
    global.cert='C:\\Users\\ulastname\\Documents\\cacert_with_doi.pem'
   ```
4. Test if pip is configured correctly

Run the command :

```
pip -v list
```

If you get no errors then you are done! 🎊

Did you get errors? If so, run the following command:

```
pip --cert  C:\\Users\\ulastname\\Documents\\cacert_with_doi.pem -v list
```

- Verify you are connected to your VPN because this command will fail if you are not
- If this command fails it might double check the correct location of the cert file is listed in your command

##### What to do if pip fails

---

1.  Error Message contains the phrase 'Open SSL Module is not available'

    ```
    Can't connect to HTTPS URL because the SSL module is not available
    ```

- **Solution:** Copy the following files from `CONDA_PATH\Library\bin` to `CONDA_PATH\DLLs`

```
libcrypto-1_1-x64.*
libssl-1_1-x64.*
```

- Make sure to copy `libcrypto-1_1-x64.*` and `libssl-1_1-x64.*` to `CONDA_PATH\DLLs`
- `CONDA_PATH` is just the path to your anaconda installation
- Example : `CONDA_PATH` : `C:\Users\sfitzpatrick\Anaconda3`
- Example : `CONDA_PATH\Library\bin`: `C:\Users\sfitzpatrick\Anaconda3\Library\bin`
- Example : `CONDA_PATH\DLLs`: `C:\Users\sfitzpatrick\Anaconda3\DLLs`
- Original [Credit](https://stackoverflow.com/questions/55185945/any-conda-or-pip-operation-give-ssl-error-in-windows-10) for this solution

#### Phase 2: Allow Conda to Work Over a Secure Network

1. Run the command

```
conda config --set ssl_verify C:\Users\uname\Documents\cacert_with_doi.pem
```

2. Verify it worked

```
conda config --show ssl_verify
```

- It should return something like

```
(coastseg) C:\Users\uname>conda config --show ssl_verify
ssl_verify: C:\Users\uname\Documents\cacert_with_doi.pem
```

##### Optional Step (only if you get ssl errors with git)

1.  Make a `.gitconfig` file

- Example location: `c:\Users\me\.gitconfig`
- Make sure to replace ` C:/Users/uname/Documents/DOIRootCA2.cer` with your username

```
[filter "lfs"]
	clean = git-lfs clean -- %f
	smudge = git-lfs smudge -- %f
	process = git-lfs filter-process
	required = true
[user]
	name = Sharon Fitzpatrick
	email = SF2309@Gmail.Com
[http]
	sslbackend = schannel
	sslVerify = true
	sslcainfo = C:/Users/uname/Documents/DOIRootCA2.cer
```

#### Phase 3: Install CoastSeg Normally

Now that pip is working, follow the rest of the installation instructions located on coastseg's front page. https://github.com/Doodleverse/CoastSeg

##### Step 2: Configure CoastSeg for a Secure Network

To use coastseg on a secure network you need to add the location of the certification file for your network in the `certifications.json` file.

1. Open the `certifications.json` file

   This file is located in the main CoastSeg directory

```
├── CoastSeg
│   ├── src
│   |  |_ coastseg
│   |  |  |_ __init__.py
│   |  |  |_bbox.py
│   |  |  |_roi.py
|
|___data
|    |_ <data downloaded here> # automatically created by coastseg when imagery is downloaded
|
|
|___certifications.json  *This is the file you need to modify*

```

![configuation_json](https://github.com/Doodleverse/CoastSeg/assets/61564689/fedfa2f7-8c83-4080-b55b-481514d4e40a)

2. Modify `cert_path` to have the full path to your cert file.

- Windows Users: Make sure to replace each `\` with `\\`

Here is an example of the full path to a cert file.

<img width="700" alt="image" src="https://github.com/Doodleverse/CoastSeg/assets/61564689/fdd61b34-b3a4-4105-be93-1284c9ec56da">

3. That's it. You're done! 🎊

# WSL SetUp Over a Secure Network

If you are using WSL to run coastseg you will need to install coastseg to your WSL instance that's because coastseg downloads files to where its installed and if its not running where it was installed it will fail to find of your data.

### Run these commands each time you open WSL

- Make sure to replace `"mnt/c/users/uname/documents/cacert_with_doi.pem"` with your user name

```
export PIP_CERT=mnt/c/users/uname/documents/cacert_with_doi.pem
export SSL_CERT_FILE=mnt/c/users/uname/documents/cacert_with_doi.pem
export REQUESTS_CA_BUNDLE=mnt/c/users/uname/documents/cacert_with_doi.pem
```

### Run these commands in WSL once

- Make sure to replace `uname` with your username `mnt/c/users/uname/documents/cacert_with_doi.pem` and verify you have `cacert_with_doi.pem` installed in that location.
- Make sure to replace `"mnt/c/users/uname/documents/cacert_with_doi.pem"` with your user name

```
conda config --set ssl_verify mnt/c/users/uname/documents/cacert_with_doi.pem
git config --global http.sslCAInfo mnt/c/users/uname/documents/DOIRootCA2.cer
```
