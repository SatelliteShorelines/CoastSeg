### How to Edit the Markdown Website

Follow these steps to edit and update the markdown website:

**Step 1:**

Switch to the `deploy_site` branch.

- You can do this using the command:
  ```bash
  git checkout deploy_site
  ```


**Step 2:**
Activate your `coastseg` environment and install `mkdocs`.

Note: Make sure you are in the main CoastSeg directory.

1. Install the required packages:
   ```
   conda activate coastseg
   cd CoastSeg 
   pip install mkdocs mkdocs-material

   ```

**Step 3:**
Visualize your changes locally

1. In the CoastSeg directory, run:
   This command generates a local version of the website so you can see your changes.

   ```
   mkdocs serve

   ```
2. Open the local version of the website by clicking the localhost link at the bottom

- Press (ctrl + S ) to save any of your changes and watch the website automatically update

   ![website_](https://github.com/user-attachments/assets/25e730c5-ea60-4679-a89c-b38af3a4c1d2)

**Step 4:**

Open the `docs` directory.

This folder contains all the markdown files that comprise the website.

**Step 5:**

Create a new markdown file or edit an existing one.

- For editing, simply modify the chosen file.
- For new files, follow **Step 6**.

**Step 6 (if creating a new file):**

Edit the `mkdocs.yml` file to include the new markdown file in the `nav` section.

- Example

```
nav:
  - Home: index.md
  - New Section:
    - Subpage: new_file.md

```

**Step 7:**

Push your changes.

Commit and push the changes, triggering a GitHub action that will update the live website.
