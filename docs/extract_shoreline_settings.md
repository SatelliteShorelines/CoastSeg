## Basic Settings

**1.Apply Cloud Mask**

- You can switch off cloud masking so that the images saved in 'CoastSeg/session/<YOUR SESSION NAME>/<ROI ID>/jpg_files/detection' won't have the clouds masked out

- Sometimes you need to turn this off because the cloud mask mistakenly identifies white sandy beaches as cloud

**Example of Masked Out Clouds**

![masked out clouds](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/461723b9-a4bc-41ec-a5b0-9ecc4eb318f7)

**Example of Masked Out Clouds in Extract Shoreline Detection Images**

- In the screenshot below see the shoreline is being broken up because the setting`dist_clouds` is removing any shorelines too close to the masked out clouds.

- Lowering `dist_clouds` or turning off the cloud mask would fix this.

- Fair warning that clouds commonly get mistaken as shorelines so generally they should be masked out

| Cloud Mask On                                                                                                          | Cloud Mask Off                                                                                                          |
| ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| ![Cloud Mask On](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/b9bae1d4-6146-49b3-b7da-e5d93b262514) | ![Cloud Mask Off](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/101e3fe1-14d3-45d7-ac3c-ab02b7fe70e8) |

**Example of Apply Cloud Mask in Settings**

![apply cloud mask](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/74ee86b6-1fd4-40d6-b237-2bca4a7da074)

**2.Dates**

- Select the start and end date to extract shorelines from

![download settings dates](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/d61a83a4-1d5e-4a23-81f0-118fb62831d9)

**3. Cloud Threshold `cloud_thresh`**

- This controls the maximum percentage of clouds allowed in the images that shorelines will be extracted from

- If an image's cloud cover exceeds this threshold a shoreline will NOT be extracted from this image

- Fair warning that clouds commonly get mistaken as shorelines so generally they should be masked out

![image](https://github.com/SatelliteShorelines/CoastSeg/assets/61564689/609f223f-c7e8-420a-9b3d-eb75421f0bbc)

**4.Reference Shoreline Buffer: `max_dist_ref`**

The `max_dist_ref` parameter defines a buffer (in meters) around the reference shoreline, limiting the extraction of shorelines to this region. Shorelines detected outside this buffer are automatically excluded.

##### Examples of Different `max_dist_ref` Settings

|                                                  ❌ Bad `max_dist_ref`                                                  |                                               ✅ Good `max_dist_ref`                                               |
| :---------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------: |
|                                                  `max_dist_ref`: 100m                                                   |                                                `max_dist_ref`:309m                                                 |
|                                           Recommend increasing buffer to 150m                                           |                                 Buffer captures entire shoreline no change needed                                  |
| ![Increased max_dist_ref](https://github.com/Doodleverse/CoastSeg/assets/61564689/62fefaac-c0c9-40c3-89b8-e433e0555f52) | ![Good max_dist_ref](https://github.com/Doodleverse/CoastSeg/assets/61564689/3ccaf660-cb22-4530-bde0-58d6f6932fa3) |

**5.Minimum Shoreline Length (min_length_sl)**
Defines the shortest length (in meters) of detected shoreline segments. Adjust this setting to filter out noise or small, irrelevant shoreline fragments.

**6.Distance from Clouds (dist_clouds)**
Specifies the minimum distance (in meters) from cloud-covered areas to avoid false shoreline detections due to cloud interference.

![dist_clouds example](https://github.com/Doodleverse/CoastSeg/assets/61564689/9fe35a16-72b1-4414-bf1a-82898d103fba)

**7.Minimum Beach Area (min_beach_area)**
Sets the smallest area (in meters²) that can be classified as shoreline, helping to distinguish between true shoreline and scattered noise.

<details>
<summary>Click to expand: How this works</summary>

During the image classification, some features (for example, building roofs) may be incorrectly labelled as sand. To correct this, all the objects classified as sand containing less than a certain number of connected pixels are removed from the sand class. The default value is 4500 m^2, which corresponds to 20 connected pixels of 15 m^2. If you are looking at a very small beach (<20 connected pixels on the images), try decreasing the value of this parameter.

</details>

![min_beach_area_coastsat_setting](https://github.com/Doodleverse/CoastSeg/assets/61564689/88f6bc9d-9ee9-42ce-98e3-e4693e8046d4)

**8.Sand Color `sand_color`**
The 'sand_color' setting is used to select the best model for classifying sand. You can choose one of the following options `default` to use the default model, `dark` for grey/black sand beaches or 'bright' for white sand beaches. This setting is only used by the coastsat shoreline extraction model not the one's used by coastseg in the `unet` notebook.

- Only change the `sand_color` parameter if you are seeing that with the default the sand pixels are not being classified as sand (in orange).
- If your beach has dark sand (grey/black sand beaches), you can set this parameter to dark and the classifier will be able to pick up the dark sand.
- On the other hand, if your beach has white sand and the default classifier is not picking it up, switch this parameter to bright. The latest classifier contains all the training data and can pick up sand in most environments (but not as accurately).

**9.Drop Intersection Points `drop_intersection_points`**
Filters out shoreline detection points that are not located on the transect they were detected on. This is off by default.

![drop_int_points](https://github.com/user-attachments/assets/48a09bc7-028a-44f8-a419-f0952dd1decd) 

**10.ROI Coverage `min_roi_coverage`**
Specifies the minimum percentage that an image must cover to be considered for downloading. This process helps to speed up the download process by skipping images that do not cover enough of the area of interest and helps to avoid image slices. Slider defaults to 50%.

![roi_coverage](https://github.com/user-attachments/assets/281ed8e5-6db0-48f8-8cd3-5abb7d85d27c)

**11.Output EPSG `output_epsg`**
Shows the current EPSG code used within CoastSeg. Defaults to EPSG 4326 (WGS84).

-Note:Once shoreline extraction is complete, output_epsg within the config.json file will be according to the rois UTM zone. 

![output_epsg](https://github.com/user-attachments/assets/eb95a46d-fc9e-4983-a7b5-68eb8e1e772a)

**Example of config.json output_epsg for Southern California (UTM Zone 11N)**

![output_epsg_config](https://github.com/user-attachments/assets/76d9a0ab-1037-4150-91c0-956df87f14e8)

**12.Check Detection `check_detection`**
Automatically set to `false` in CoastSeg. If set to 'true' the user can control each shoreline detection interactively to accept ot reject each shoreline.

**13.Adjust Detection `adjust_detection`**
Automatically set to `false` in CoastSeg. If set to 'true' the user can manually adjust the threshold used to map the shoreline on each image.

**14.Save Figure `save_figure`**
Automatically set to `true` in CoastSeg. Saves a figure of each mapped shoreline to the session detection folder. 
Example path to detection folder: CoastSeg\sessions\sample_session1\ID_zyh1_datetime06-11-24__03_02_55\jpg_files\detection

![save_figure](https://github.com/user-attachments/assets/ecfeeead-97a8-401c-836e-e2690afc141b)

## Advanced Settings (Advanced Users only)

---

The advanced settings are used to determine where each shoreline vector intersects each transects. You generally shouldn't need to modify these settings, but if you have a complex shoreline you might want to.

**1.Minimum Number of Shoreline Points `min_points`**

Sets the threshold for the minimum number of shoreline points required to identify a valid shoreline intersection with the transect. If the number of points is below this threshold, the function will return NaN (Not a Number) for that intersection.

- ** Reduce `min_points`:** If your shoreline is complex and scattered, you might need to reduce this number to ensure that some intersections can be calculated. However, be cautious because a lower number might lead to less reliable intersections.

- ** Increase `min_points`:** If you have have ample and dense shoreline data points and want to make sure that the intersection is calculated based on a substantial number of points for robustness you might want to increase this value. This could be the case for well-sampled or highly detailed shoreline datasets.

**2.Alongshore Distance `along_dist`**

Defines the maximum alongshore distance (in meters) from the transect within which shoreline points are considered for calculating intersections.

- Increase: If your shoreline is more complex and irregular, you may need to increase this value to capture more points for intersection calculation. However, be aware that larger values might also include more irrelevant points, so it's about finding a balance.

- Decrease: Decreasing this parameter might be useful when the shoreline is simple, straight, and without significant alongshore variability. A smaller along_dist value would restrict the intersection calculation to points closer to the transect, reducing potential noise from farther points.

**3.Maximum Standard Deviation `max_std`**

Limits the variability of shoreline points used to compute the median intersection, reducing the influence of outliers.

`max_std` is the maximum acceptable standard deviation (in meters) for the shoreline points when calculating the median intersection. If the standard deviation of the points is above this value, the function will return NaN if in 'nan' mode or if in 'max' mode the maximum intersection will be returned.

- Increase : A complex shoreline with a lot of variability might necessitate a higher `max_std` value. However, a larger `max_std` might also lead to the inclusion of more outliers, so use this parameter carefully.

- Decrease : You might want to decrease this value when the intersections are based on a group of shoreline points that are close to each other in the cross-shore direction, ensuring a more precise median intersection. This could be beneficial when the shoreline data are of high quality with limited cross-shore variability.

**4.Maximum Range `max_range`**

Specifies the maximum range (in meters) allowed for the shoreline points when calculating the median intersection. If the range is larger than this, the function may return NaN or use the maximum intersection, depending on other settings.

- **Increase:** For a complex, irregular shoreline, a larger `max_range` might be required. However, as with `max_std`, a larger `max_range` could also introduce more dispersion in the intersection points.

- **Decrease:** Similar to 'max_std', you might want to decrease this value to ensure a more precise median intersection when the shoreline data are of high quality with limited cross-shore range. Decreasing 'max_range' will exclude intersections that are based on points with a large range in the cross-shore direction.

**5.Minimum Chainage `min_chainage`**

The furthest landward distance (in meters) from the transect origin that an intersection is accepted. Beyond this point, the function returns NaN for the intersection.

- **Increase :** If your shoreline has a lot of landward variation or recesses, you might need to increase this value. However, this could also increase the risk of including irrelevant points landward of the actual shoreline.

- **Decrease :** Decreasing this parameter might be useful when the shoreline does not have significant landward recesses and users want to exclude points that are too far landward of the transect origin. This could help avoid irrelevant points from impacting the intersection calculation.

**6.Percentage Multiple `prc_multiple`**

In 'auto' mode, this setting decides when to switch from NaN (indicating no reliable intersection) to the maximum intersection value based on shoreline data variability.

- If the percentage of data points where the standard deviation is larger than `max_std` is above this value, the function will switch to returning the maximum intersection.

- If your shoreline is complex and you're using the `auto` mode, you might need to adjust this value based on the specific dispersion characteristics of your data.

**7.Multiple Intersections Mode `multiple_inter`**

Manages scenarios where multiple potential shoreline intersections occur along a single transect.

This is quite common, for example when there is a lagoon behind the beach and the transect crosses two water bodies. The function will analyze transect by transect to identify these kinds of cases and depending on the selected mode choose to:

- **'nan':**

  -- Always assign a NaN when there are multiple intersections.

- **'max':**

  -- Always take the max (intersection the furthest seaward).

- **'auto':**

  -- Switch from returning NaN to returning the maximum intersection if a certain percentage of data points have a standard deviation larger than `max_std`.

- If your shoreline is complex, which means the intersections have high dispersion (i.e., large standard deviation or range) `auto` mode should be beneficial as it automatically decides whether to use NaN or the maximum intersection based on the percentage of points exceeding `max_std`.

- If 'auto' is chosen, the `prc_multiple` parameter will define when to use the max, by default it is set to 0.1, which means that if 10% of the time-series show multiple intersections, then the function thinks there are two water bodies.
