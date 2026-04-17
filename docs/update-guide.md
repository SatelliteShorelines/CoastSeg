# Update CoastSeg

Use this guide to update CoastSeg in the environment you already use:

- Conda install
- Pip install
- Pixi install
- Local git clone

If you are not sure which method you used, check the commands you normally run to start CoastSeg:

- `conda activate coastseg` usually means a Conda environment
- `pip install coastseg` usually means a pip environment
- `pixi shell` means a Pixi environment backed by your local git clone

!!! note
   Pixi installs CoastSeg from your local git clone. If you use Pixi, updating the git clone is part of updating CoastSeg.

## Before You Start

- Open a terminal.
- If you use Conda or Miniforge, make sure Conda is available in that terminal.
- Decide what you want to update:
   - Only the installed CoastSeg package
   - Your local CoastSeg git clone
   - Both the package and the git clone

## Choose Your Update Path

| If you installed CoastSeg with... | Update this way | Also update your git clone? |
| --- | --- | --- |
| `conda install coastseg` | [Update a Conda Environment](#update-a-conda-environment) | Only if you also work from a local clone |
| `pip install coastseg` | [Update a Pip Environment](#update-a-pip-environment) | Only if you also work from a local clone |
| `pixi shell` | [Update a Pixi Environment](#update-a-pixi-environment) | Yes. Pixi uses your local clone |
| `git clone ...CoastSeg.git` | [Update a Git Clone](#update-a-git-clone) | Yes |

## Update a Git Clone

Use this section if you want the latest notebooks, scripts, docs, or source code from GitHub.

This is required for Pixi installs and optional for Conda or pip installs.

**Go to your CoastSeg clone**

```bash
cd <coastseg_location>
```

**Check that `origin` points to the CoastSeg repository**

```bash
git remote -v
```

You should see something like:

```text
origin  https://github.com/SatelliteShorelines/CoastSeg.git (fetch)
origin  https://github.com/SatelliteShorelines/CoastSeg.git (push)
```

If `origin` is missing, add it:

```bash
git remote add origin https://github.com/SatelliteShorelines/CoastSeg.git
```

**Check for local changes**

```bash
git status
```

If Git says your working tree is clean, pull the latest changes:

```bash
git pull origin main
```

**If `git pull` fails because you changed local files**

If you want to keep your local changes, stash them first:

```bash
git stash
git pull origin main
```

(OPTIONAL) If you want to keep your local changes, stash them first, then retrieve them from the stash.


```bash
git stash pop
```

🎉 You're done!

**Last resort: force your clone to match GitHub**

Only use this if you want to discard local changes in the clone.

!!! warning
   This will discard local changes in the clone. If you want to keep anything, copy it somewhere safe first. This is especially important for files like `certifications.json` or any notebooks, scripts, or settings you edited locally.

```bash
git fetch origin
git reset --hard origin/main
```

## Update a Conda Environment

Use this section if you installed CoastSeg from Conda or conda-forge.

**Activate the environment**

```bash
conda activate coastseg
```

**Update to the latest CoastSeg version**

```bash
conda update coastseg
```

**Install a specific CoastSeg version**

Replace `<version>` with the version you want, for example `1.2.3`.

```bash
conda install coastseg=<version>
```

**Verify the update**

```bash
python -c "import coastseg; print('CoastSeg import OK')"
```

## Update a Pip Environment

Use this section if you installed CoastSeg with `pip install coastseg`.

**Activate the environment**

If you use Conda to manage the Python environment, activate it first:

```bash
conda activate coastseg
```

If you use a virtual environment instead, activate that environment before running pip.

**Update to the latest CoastSeg version**

```bash
pip install --upgrade coastseg
```

**Install a specific CoastSeg version**

Replace `<version>` with the version you want, for example `1.1.26`.

```bash
pip install coastseg==<version>
```

**About pip dependency warnings**

You may see a message like this during the update:

```text
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
```

This warning is common in older Python environments. If CoastSeg imports correctly after the update, the installation may still be usable.

**Verify the update**

```bash
python -c "import coastseg; print('CoastSeg import OK')"
```

## Update a Pixi Environment

Use this section if you start CoastSeg with `pixi shell`.

!!! warning
   Do not run `pip install` or `conda install` inside a Pixi environment.

- Pixi installs CoastSeg from your local git clone
- Updating the clone is part of updating CoastSeg
- Pixi environments should be updated with `git pull`, `pixi install`, and `pixi shell`

**Go to your CoastSeg clone**

```bash
cd <coastseg_location>
```

Make sure this folder contains `pyproject.toml` and `pixi.lock`.

**Pull the latest code**

```bash
git pull origin main
```

If your clone uses a different default branch, replace `main` with that branch name.

**Refresh the Pixi environment**

For the default environment:

```bash
pixi install
pixi shell
```

If you use the zoo workflow or another named Pixi environment, activate that environment after the install:

```bash
pixi install
pixi shell -e all
```

**Verify the update**

```bash
python -c "import coastseg; print('CoastSeg import OK')"
```

**If the Pixi environment looks broken**

If you previously ran `pip install` or `conda install` inside the Pixi environment and now get import errors, rebuild it from the repo root:

```bash
pixi clean
pixi install
```

If that still does not fix the issue, remove the `.pixi` folder and recreate the environment:

```bash
pixi install
```

## Recommended Update Workflows

**I installed CoastSeg with Conda**

```bash
conda activate coastseg
conda update coastseg
python -c "import coastseg; print('CoastSeg import OK')"
```

**I installed CoastSeg with pip**

```bash
conda activate coastseg
pip install --upgrade coastseg
python -c "import coastseg; print('CoastSeg import OK')"
```

**I installed CoastSeg with Pixi**

```bash
cd <coastseg_location>
git pull origin main
pixi install
pixi shell
python -c "import coastseg; print('CoastSeg import OK')"
```

**I only want the latest notebooks and source code from GitHub**

```bash
cd <coastseg_location>
git pull origin main
```

## Verify Everything Worked

These checks are usually enough:

```bash
python -c "import coastseg; print('CoastSeg import OK')"
git status
```

Expected git output after a successful clone update:

```text
On branch main
Your branch is up to date with 'origin/main'.
```

If you still have update problems after following the matching section above, re-check that you updated the same environment you actually use to run CoastSeg.
