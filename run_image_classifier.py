# Script to run the image classifier on a folder of images
# How to use:
# 1. Find the folder that contains the RGB images you want to classify.
#   Set input_path to the full path of this folder.
#   This is usually something like:
#       Coastseg/data/ID_xxx_datetimexxx/jpg_files/preprocessed/RGB
# 2. Run the script
# 3. After it finishes:
#    - Open the folder you set as input_path.
#    - You should see:
#        - A subfolder called "bad" containing images classified as bad.
#        - A file called "classification_results.csv" with all the classification results.
from coastseg import classifier
import os

#  Enter the full path to the folder containing the images to be classified (aka the RGB folder in Coastseg/data/ID_xxx_datetimexxx/jpg_files/preprocessed/RGB)
input_path = r""
output_path = input_path
output_csv = os.path.join(input_path, "classification_results.csv")

# classifier_path = classifier.get_image_classifier('RGB')
classifier_path = classifier.get_image_classifier("rgb")
print(f"Classifier path: {classifier_path}")
classifier.run_inference_rgb_image_classifier(
    classifier_path, input_path, output_path, output_csv, threshold=0.40
)
