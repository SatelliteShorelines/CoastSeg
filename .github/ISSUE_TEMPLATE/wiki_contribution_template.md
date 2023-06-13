---
name: Wiki contribution
about: Write a contribution to the wiki
title: ''
labels: ''
assignees: ''

---

Thanks for submitting a contribution to the wiki! Please read and follow these instructions carefully, then delete this introductory text to keep your issue easy to read. Feel free to keep the titles described here to ensure our team can easily understand your issue. Note that the issue tracker is NOT the place for usage questions and technical assistance.

## Describe the Contribution
Briefly describe in a sentence what you are contributing and which part of the wiki you would like your contribution to be added to. If you are changing an existing part of the wiki please copy the original section and include the modified section below the original. Alternatively, if you are submitting a new page to the wiki please use the rest of this issue to write your wiki page as it would appear on github.

## Example Contribution Title
Example contribution to the wiki. In this example issue the wiki section 'How to Download Imagery' will be modified. 

**Wiki Title : How to Download Imagery**
### Original
2. Draw a bounding box along a coastline.
- Make sure to click the load shorelines button to check if shorelines exist within the bounding box.
- ⚠️ If no shorelines can be loaded within the bounding box then ROIs cannot be created.


### Modification

2. Draw a bounding box along the coastline.
Using the rectangle tool in the righthand corner of the map draw a bounding box around the region you want to create regions of interest(ROI) within. Do not draw the bounding box that's too large otherwise it will be removed from the map.
- Before clicking 'Generate ROIs' click the load shorelines button to check if shorelines exist within the bounding box.
- ⚠️ If no shorelines can be loaded within the bounding box then ROIs cannot be created. If no shorelines are available in that region, try uploading your own shorelines from a geojson file.
- Its also a good idea to check if any transects exist in your bounding box as well. If no transects are available in that region, try uploading your own transects from a geojson file.


## Screenshots (If applicable)
If applicable, add screenshots that will be used in the wiki



## ALL software version info
**Desktop (please complete the following information):**
(this library, plus any other relevant software, e.g. bokeh, python, notebook, OS, browser, etc)
 - OS: [e.g. iOS]
 - Browser [e.g. chrome, safari]
 - Version [e.g. 22]


### **Additional context**
Add any other context about the contribution here.