### How to Edit the Markdown Website

Follow these steps to edit and update the markdown website:

**Step 1:**

Switch to the `deploy_site` branch.

- You can do this using the command:
  ```bash
  git checkout deploy_site
  ```

**Step 2:**

Open the `docs` directory.

This folder contains all the markdown files that comprise the website.

**Step 3:**

Create a new markdown file or edit an existing one.

- For editing, simply modify the chosen file.
- For new files, follow **Step 4**.

**Step 4 (if creating a new file):**

Edit the `mkdocs.yml` file to include the new markdown file in the `nav` section.

- Example

```
nav:
  - Home: index.md
  - New Section:
    - Subpage: new_file.md

```

**Step 5:**

Visualize your changes locally

1. Install the required packages:

   ```
   pip install mkdocs mkdocs-material

   ```

2. In the CoastSeg directory, run:
   This command generates a local version of the website so you can see your changes.

   ```
   mkdocs serve

   ```

**Step 6:**

Push your changes.

Commit and push the changes, triggering a GitHub action that will update the live website.
