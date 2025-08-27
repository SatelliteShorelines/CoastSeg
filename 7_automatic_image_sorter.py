from coastseg import classifier

# Disclaimer this script will not work without having installed tensorflow 2.12


# This script will automatically sort images in a directory using the model. 
# The bad images are moved to a subdirectory called 'bad'.
# It is meant be used on CoastSeg/data/<ROI NAME>/jpg_files/preprocessed/RGB

# select the RGB directory for an ROI from /data
input_directory =r""

# run the classifier to automatically sort the images
classifier.sort_images_with_model(input_directory,threshold=0.40)
