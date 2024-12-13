from coastseg import classifier
import os

input_path =r'C:\development\doodleverse\coastseg\CoastSeg\sessions\coreg_session2\good'
output_path = input_path
output_csv=os.path.join(input_path,'classification_results.csv')

segmentation_classifier = classifier.get_segmentation_classifier()
classifier.run_inference_segmentation_classifier(segmentation_classifier,
                input_path,
                output_path,
                output_csv,
                threshold=0.40)



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