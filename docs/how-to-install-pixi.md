## Install CoastSeg with Pixi

**Pixi warning (please read)**
- Keep your local CoastSeg repo **up to date**, or you will use an **outdated** CoastSeg.
- This is an **editable install**: Pixi builds CoastSeg from the code in your **local git clone**.
- ⚠️⚠️⚠️**Do not run `pip install` or `conda install` inside a Pixi environment.** It can permanently break the environment.⚠️⚠️⚠️
  - Usually this is fine but if you start getting import errors use the advice below
  - If that happens: delete the `.pixi/` folder and re-run `pixi shell --locked` to reinstall all the dependencies

## Table of Contents

- [Before anything: Install Pixi](#before-anything-install-pixi)
- [Basic Pixi install (recommended)](#basic-pixi-install-recommended)
- [Use Pixi without admin access](#use-pixi-without-admin-access)
- [Activate an existing Pixi environment](#activate-an-existing-pixi-environment)
- [Upgrade your Pixi environment](#upgrade-your-pixi-environment-get-the-latest-coastseg)
- [Pixi Command Reference Table](#pixi-command-reference-table)
- [FAQs](#faqs)
---


### Before anything:  Install Pixi
Visit the [Pixi installation page](https://pixi.sh/latest/) and follow instructions to install Pixi in your shell.
    
## Basic Pixi install (recommended)

1. **Clone the CoastSeg repo (shallow clone)**
    - You only need to do this once
   ```bash
   git clone --depth 1 https://github.com/SatelliteShorelines/CoastSeg.git
    ```
2. Navigate to the CoastSeg Directory

    - Make sure the directory you are at contains the files. You can use the `ls` command to list the files.
        - `pyproject.toml`
        - `pixi.lock`

    ```
    cd CoastSeg
    # Make sure this folder contains pyproject.toml
    ```
3. Create + activate the Pixi environment
    - This is like  `conda install` + `conda activate` in one step, it reads  CoastSeg’s `pixi.lock` file to build the environment and then activates it
    ```
    pixi shell --frozen
    ```
4. Verify it worked

    - You should see something like (coastseg:all) in your terminal prompt.

    - Test the import:
    ```
    python -c "import coastseg; print('CoastSeg import OK')"

    ```
![pixi default env](https://github.com/user-attachments/assets/83920c04-1182-479f-9bcc-ba97a1a7ad33)

5. Exit the Environment (optional)

Exit the current Pixi environment:

```
exit

```
- notice how `(coastseg)` is no longer in front, this means that we have exited the coastseg environment

![exit coastseg pixi](https://github.com/user-attachments/assets/d52aa046-e734-405c-ac01-9ca413c68942)


## Use Pixi without admin access
If you cannot install Pixi system-wide, install it into a conda environment:

    ```
    conda create -n coastseg_pixi python=3.10 -y
    conda activate coastseg_pixi
    conda install -c conda-forge pixi -y
    ```
Then follow the steps in [Basic Pixi install](#basic-pixi-install-recommended) (starting from git clone).

## Activate an existing Pixi environment
- Activate your pixi environment using the `pixi shell` command
- Use `pixi shell -e all` if you want to use the zoo workflow.

    ```
    cd <coastseg_location>
    pixi shell
    ```

## Upgrade your Pixi environment (get the latest CoastSeg)

> ❗ Reminder (Pixi installed via conda): If you installed Pixi inside > a conda environment (for example coastseg_pixi), activate that conda > environment before running any pixi commands:

```
conda activate coastseg_pixi
```

1. Go to your CoastSeg repo

    ```
    cd <coastseg_location>
    ```
2. Set the remote connection to CoastSeg on GitHub (if needed)

    - A `git remote` is a URL pointing to the CoastSeg GitHub repo, used to fetch and pull the latest changes.

    - `origin` is just the name of that remote — it could be named anything (like cat).

    ```
    git remote add origin https://github.com/Doodleverse/CoastSeg.git
    ```

    Verify the remote is set up:
    ```
    git remote -v
    ```
    You should see something like
    ```
    origin  https://github.com/SatelliteShorelines/CoastSeg (fetch)
    origin  https://github.com/SatelliteShorelines/CoastSeg (push)

    ```
3. Pull the latest changes

    ```
    git pull origin
    ```
    If it fails due to local changes:
    ```
    git stash
    git pull origin
    ```
4. Re-create/update the environment from the updated lockfile
    
    ```
    pixi shell --frozen
    ```

5. Verify the upgrade

    ```
    python -c "import coastseg; print('CoastSeg import OK')"
    ```

# Pixi Command Reference Table

| Command | Description | Conda Equivalent | Documentation |
| --- | --- | --- | --- |
| `pixi shell -e <NAME>` | Activate Pixi environment named `<NAME>` | `conda activate <NAME>` | [Pixi shell docs](https://pixi.sh/latest/features/multi_environment/#user-interface-environment-activation) |
| `exit` | Exit the current Pixi environment | `conda deactivate` | [Pixi exit docs](https://pixi.sh/latest/switching_from/conda/#key-differences-at-a-glance) |
| `pixi install` | Install dependencies from `pyproject.toml` and update `pixi.lock` | `conda install` | [Pixi install docs](https://pixi.sh/latest/tutorials/python/#installation-pixi-install) |
| `pixi install --frozen` | Install dependencies strictly from `pixi.lock` without updating it, even if it differs from `pyproject.toml` | Install from a conda-lock file | [Pixi frozen install docs](https://prefix.dev/docs/pixi/cli#install) |

---

## FAQs

### Why can't I use pip install or conda install inside my pixi environment?
Pixi is supposed to managed your dependencies and if you run `pip install` or `conda install` those packages get written to a different location on your computer than where your pixi environments get saved. This means when you run `pixi shell` it won't be able to find any of the dependencies you installed with `pip install` or `conda install`.

### What is `pixi.lock`?

The `pixi.lock` file explicitly lists exact versions of Conda and PyPI packages required for each environment defined in `pyproject.toml`. You can open this file to view detailed package version information and sources (PyPI or Conda Forge).

### Wait, what happened to `pyproject.toml`?

The `pyproject.toml` file remains mostly unchanged, but now includes additional sections to help Pixi configure your Python environments more effectively.

### Hold up, what do you mean there are multiple environments?
Yeah, that confused me too at first. What's special about pixi is that it can define multiple related environments all within the same file, `pyproject.toml`. 


### Why did you separate the environments like this?
Great question, I did it because for our zoo workflow we require tensorflow version 2.12 to run our models correctly, which isn't a package avavilable on conda forge for windows machines. Since we want CoastSeg to be available in on conda-forge I opted to make it an optional dependency and a separate environment. But don't worry you can still the  third environment called `all` to be able to use the coastsat and zoo workflows at the same time with the `pixi shell -e all` command.