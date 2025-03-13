

## Non-Windows Users

Visit the [Pixi installation page](https://pixi.sh/latest/) and follow instructions to install Pixi in your shell.

## Tutorial: Install Pixi with Powershell for Windows

### 1. Install Pixi
Visit the [Pixi installation page](https://pixi.sh/latest/) and follow instructions to install Pixi in your shell.

### 2. Open Powershell and Configure Pixi
This code below tells Powershell to allow pixi to connect to the shell
```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
pixi shell
```
<details>
<summary>Technical explanation</summary>
This sets the current user's PowerShell execution policy to "RemoteSigned," allowing local scripts to run unsigned while requiring signatures for downloaded scripts.
- Local scripts run without needing a digital signature, allowing for flexibility during development.
- Scripts from the internet must be signed, which helps protect against running untrusted or tampered code.
</details>
<details>
<summary>Command Breakdown</summary>
        - `Set-ExecutionPolicy`: Changes the PowerShell script execution policy.</br>
        - `RemoteSigned`: Allows unsigned local scripts, but requires a signature for downloaded ones.</br>
        - `-Scope CurrentUser`: Affects only your user account (no admin rights needed).
</details>

### 3. Move to CoastSeg
Change directories to the location of your pyproject.toml

1. `cd <CoastSeg folder containing pyproject.toml>`

2. Check if the `pyproject.toml` exists in that directory with `Test-Path .\pyproject.toml` 
    1. If it exists it will print `True`

### 4. Validate Pixi is Working
Activate the environment with:

```
pixi shell --frozen

```

1. This will activate the default environment, which in this example is called `(coastseg)`

![pixi shell frozen](https://github.com/user-attachments/assets/c2b459b5-082a-4614-88e7-15afa7de44c9)

If you get an error like:

![pixi permission_error](https://github.com/user-attachments/assets/38da7a21-de80-4e10-b278-937c1358e203)

Then try the following (you don’t need admin permissions)

- This command tells powershell that Pixi is safe to connect to Powershell, buts temporary so you will need to re-run it each time you run `pixi shell` in a new powershell window

```powershell
function Invoke-Pixi {
    powershell.exe -ExecutionPolicy Bypass -Command "pixi $args"
}

Set-Alias pixi Invoke-Pixi -Option AllScope

pixi shell --frozen
```
    
## Install CoastSeg with Pixi

### Step 1: Navigate to Project Directory

Open your preferred shell (this example uses PowerShell) and navigate to your project's directory containing:

- `pyproject.toml`
- `pixi.lock`

### Step 2: Install the Environment

Install dependencies specified in the `pixi.lock` file:

```
pixi install --frozen

```

![pixi install frozen](https://github.com/user-attachments/assets/83d60c87-aaeb-4d3c-b146-96670cf378f6)

### Step 3: Activate the Default Environment

Activate the environment with:

```
pixi shell --frozen

```

1. This will activate the default environment, which in this example is called `(coastseg)`

![pixi shell frozen](https://github.com/user-attachments/assets/c2b459b5-082a-4614-88e7-15afa7de44c9)

If you get an error like:

![pixi permission_error](https://github.com/user-attachments/assets/38da7a21-de80-4e10-b278-937c1358e203)

Then try the following (you don’t need admin permissions)

- this command tells powershell that Pixi is safe to connect to Powershell

```powershell
function Invoke-Pixi {
    powershell.exe -ExecutionPolicy Bypass -Command "pixi $args"
}

Set-Alias pixi Invoke-Pixi -Option AllScope

pixi shell --frozen
```

### Step 4: Verify Installation

Check that the environment is correctly set up by running:

```
python -c "import coastseg"

```

![pixi default env](https://github.com/user-attachments/assets/83920c04-1182-479f-9bcc-ba97a1a7ad33)

### Optional: Open the CoastSat notebook

- If you don't want to use the zoo workflow and only want to use the CoastSat workflow then use the command below to run the `SDS_coastsat_classifier.ipynb` notebook in the environment.

```
jupyter lab SDS_coastsat_classifier.ipynb

```


### Step 5: Exit the Environment

Exit the current Pixi environment:

```
exit

```

- notice how `(coastseg)` is no longer in front, this means that we have exited the coastseg environment

![exit coastseg pixi](https://github.com/user-attachments/assets/d52aa046-e734-405c-ac01-9ca413c68942)


### Step 6: Activate a Custom Environment (Zoo Workflow)

Activate the Pixi environment configured for machine learning workflows (e.g., Zoo workflow):

```
pixi shell -e ml

```

This environment adds TensorFlow and Transformers, which are essential for running Zoo workflow custom models.

![pixi shell ml](https://github.com/user-attachments/assets/3d26ed7e-fe7f-49cd-b5c5-d9e0e2287bf6)


### Step 7: Verify ML Environment

In the `ml` environment, verify that TensorFlow and Transformers are installed:

```
python -c "import tensorflow; from transformers import TFSegformerForSemanticSegmentation;"

```



## Command Reference Table

| Command | Description | Conda Equivalent | Documentation |
| --- | --- | --- | --- |
| `pixi shell -e <NAME>` | Activate Pixi environment named `<NAME>` | `conda activate <NAME>` | [Pixi shell docs](https://pixi.sh/latest/features/multi_environment/#user-interface-environment-activation) |
| `exit` | Exit the current Pixi environment | `conda deactivate` | [Pixi exit docs](https://pixi.sh/latest/switching_from/conda/#key-differences-at-a-glance) |
| `pixi install` | Install dependencies from `pyproject.toml` and update `pixi.lock` | `conda install` | [Pixi install docs](https://pixi.sh/latest/tutorials/python/#installation-pixi-install) |
| `pixi install --frozen` | Install dependencies strictly from `pixi.lock` without updating it, even if it differs from `pyproject.toml` | Install from a conda-lock file | [Pixi frozen install docs](https://prefix.dev/docs/pixi/cli#install) |

---

## FAQs

### What is `pixi.lock`?

The `pixi.lock` file explicitly lists exact versions of Conda and PyPI packages required for each environment defined in `pyproject.toml`. You can open this file to view detailed package version information and sources (PyPI or Conda Forge).

### Wait, what happened to `pyproject.toml`?

The `pyproject.toml` file remains mostly unchanged, but now includes additional sections to help Pixi configure your Python environments more effectively.

### Hold up, what do you mean there are multiple environments?
Yeah, that confused me too at first. What's special about pixi is that it can define multiple related environments all within the same file, `pyproject.toml`. For CoastSeg I've created two environments, one thats the default environment that can run the coastsat workflow and the other environment called `ml` that adds `tensorflow` & `transformers` to the environment to run the zoo workflow. If you are only interested in the coastsat workflow then you will only need the default environment.


### Wait, I want to use both workflows and I don't want to switch. 
Yeah, I'm lazy too that's why I created a third environment called `all` that contains the dependencies for both environments. To use this simply use `pixi shell -e all` and this will open a pixi shell that can use both workflows. Once the shell opens you can use python commands, jupyter commands just like you would with conda prompt.

### Why did you separate the environments like this?
Great question, I did it because for our zoo workflow we require tensorflow version 2.12 to run our models correctly, which isn't a package avavilable on conda forge for windows machines. Since we want CoastSeg to be available in on conda-forge I opted to make it an optional dependency and a separate environment. But don't worry you can still the secret third environment called `all` to be able to use the coastsat and zoo workflows at the same time with the `pixi shell -e all` command.