# How to Release a New Package 

---

CoastSeg has a github action that publishes a new pypi package if the commit is tagged with the version number of the package to release on pypi.
To ensure a consistent and smooth package release process, follow this step-by-step guide. The release procedure centers around tagging the commit properly.


## Release Steps

1. Ensure your local branch is up-to-date with the main branch.

    ```bash
    git pull origin main
    ```

2. Commit your all changes.

    ```bash
    git commit -m "<explain changes"  
    ```

3. Modify the `pyproject.toml` file with the new version number

   - Under the `[project]` section change the `version = ""` to the new version

   - Example: to release version 1.2.6 I would write `version = "1.2.6"`

   ```
      [project]
   name = "coastseg"
   dynamic = ["readme"]
   version = "1.2.5"   # change this line to your new version
   authors = [
   { name=" Sharon Fitzpatrick", email="sharon.fitzpatrick23@gmail.com" },
   ]
   ```

4. Commit the modified `pyproject.toml` file.

   - In this commit only include the changed `pyproject.toml` file

   - This makes is easy to track different versions

    ```bash
    git commit -m "Release v1.0.3"  # Replace with your version number
    ```

5. To tag the most recent commit, simply use:

   - See the tagging format guide below to learn how to format the tags

   - The expected format is `v<version_number>` so for version 1.0.3 it would be `v1.0.3`

    ```bash
    git tag v1.0.3
    ```

6. Alternatively, if you need to tag a specific commit (other than the most recent one), first find the commit's hash using:

    ```bash
    git log --oneline
    ```

    This will display a list of recent commits with their shortened hash. Identify the commit you want to tag, then tag it using:

    ```bash
    git tag v1.0.3 COMMIT_HASH  # Replace with your version number and the appropriate commit hash
    ```

7. Push the commit to the repository.

    - This will trigger tests to run automatically in the Actions tab in the CoastSeg repository.

    ```bash
    git push origin main
    ```

8. Push the tag to the repository.

    - This will trigger an action called "Publish Package to PyPi", which will automatically release the package on pypi, then later trigger the conda package to be released.

    ```bash
    git push origin v1.0.3  # Replace with your version number
    ```

9. Release the Conda Package

    - Visit the CoastSeg [conda-forge feedstock](https://github.com/conda-forge/coastseg-feedstock)

    - The bots at the CoastSeg conda-forge feedstock will scan pypi for the new coastseg release and when it finds a new release it will open a pull request with the new version number

    - Merge the new Pull Request on the conda forge feedstock and the bots will automatically release a new version

   
   ![how to release a conda package merge pull request](https://github.com/user-attachments/assets/b096e3b2-f200-4ca8-932b-05f511e79226)

## Tagging Format

When you're ready to release a new package, push a commit with a tag that matches one of the following formats:

- Major, minor, and patch versions: `v[0-9]+.[0-9]+.[0-9]+`

  - Example: `v1.0.3`

- Alpha versions: `v[0-9]+.[0-9]+.[0-9]+a[0-9]+`

- Beta versions: `v[0-9]+.[0-9]+.[0-9]+b[0-9]+`

- Release candidate versions: `v[0-9]+.[0-9]+.[0-9]+rc[0-9]+`

- Development versions: `v[0-9]+.[0-9]+.[0-9]+dev[0-9]+`
  - Example: `v1.2.0dev1`

### ✅ Good Tag Names

- v1.0.3
- v1.2.0dev1

### ❌ Bad Tag Names

- 1.2.0 : Missing the "v"