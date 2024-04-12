# Contributing Workflows

To contribute your own workflows to CoastSeg you will need to create your own jupyter notebook, test it and then submit a Pull Request to CoastSeg. The CoastSeg team will examine your pull request, test it, then merge your workflows. Workflows should be self contained and not impact other workflows. That being said contributions to improve other workflows are welcome, but please submit improvements as separate Pull Requests instead of with workflow contributions. Once your workflow is accept you will be responsible for maintaining the workflow and if your workflow is no longer maintained the CoastSeg team may drop support for your workflow.

1. Create a notebook with the name of your workflow

- Example: `SDS_new_workflow_name.ipynb` in CoastSeg

2. Create a new python file in `CoastSeg/src/coastseg` with the code needed to run your workflow

- Example: Create `new_workflow_name.py` which contains all the code for your workflow

3. Submit a Pull Request with your workflow
