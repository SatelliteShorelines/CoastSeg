---
name: 'Issue: Data Contribution'
about: Contribute a shoreline, transect, or other data
title: ''
labels: ''
assignees: ''

---

Thanks for contributing! Please read and follow these instructions carefully, then delete this introductory text to keep your issue easy to read. Feel free to keep the titles described here to ensure our team can easily integrate your contribution.

## Describe the Contribution
A clear and concise description of what you are contributing. 
- Where is the data from?
- What is its purpose? 
- How large is it (MB)?

### Data Type (Choose One)
1. Shoreline geojson file
2. Transect geojson file
3. Other 

#### Shoreline Contributions
You must include a json file called `file_bounds.json` that includes a bounding boxes for each of the shorelines you wish to contribute. These bounding boxes are used to check if the user's ROI's intersect with the shoreline. You can check [file_bounds.json](https://zenodo.org/record/7033367#.YxFQFXbMI2w) for an example.

The shorelines must be contained within geojson files that are no larger than 15MB. You can check [zenodo release for USA county shorelines](https://zenodo.org/record/7033367#.YxFQFXbMI2w) for an example.

#### Transect Contributions
The transects must be contained within geojson files that are no larger than 15MB. 


## Screenshots
If applicable, add screenshots to show your contribution. For example for contributing a shoreline show where the shoreline is on a map.


### **Additional context**
Add any other context about the contribution here.

## Data
Upload your file(s) as a zip or link to a safe location to download the data from.
