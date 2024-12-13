from coastseg import classifier
import os

input_path =r'C:\development\doodleverse\coastseg\CoastSeg\data\ID_1_datetime06-04-24__12_09_54\jpg_files\preprocessed\RGB'
output_path = input_path
output_csv=os.path.join(input_path,'classification_results.csv')

# classifier_path = classifier.get_image_classifier('RGB')
classifier_path = classifier.get_image_classifier('rgb')
print(f"Classifier path: {classifier_path}")
classifier.run_inference_rgb_image_classifier(classifier_path,
                input_path,
                output_path,
                output_csv,
                threshold=0.40)

# try the gray
# classifier_path = classifier.get_image_classifier('gray')
# print(f"Classifier path: {classifier_path}")
# classifier.run_inference_gray_image_classifier(classifier_path,
#                 input_path,
#                 output_path,
#                 output_csv,
#                 threshold=0.40)




# apply good bad classifier to the downloaded imagery
# for key in roi_settings.keys():
#     data_path = os.path.join(roi_settings[key]['filepath'],roi_settings[key]['sitename'])
#     RGB_path = os.path.join(data_path,'jpg_files','preprocessed','RGB')
#     print(f"Sorting images in {RGB_path}")
#     input_path =RGB_path
#     output_path = RGB_path
#     output_csv=os.path.join(RGB_path,'classification_results.csv')
#     # model_path = os.path.join(r'C:\development\doodleverse\coastseg\CoastSeg\src\coastseg\classifier_model','best.h5')
#     model_path = classifier.get_classifier()
#     classifier.run_inference(model_path,
#                 input_path,
#                 output_path,
#                 output_csv,
#                 threshold=0.10)