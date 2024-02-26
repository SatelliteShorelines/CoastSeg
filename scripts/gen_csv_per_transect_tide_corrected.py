import os 
import pandas as pd

def save_csv_per_id(
    df: pd.DataFrame,
    save_location: str,
    filename: str = "timeseries_tidally_corrected.csv",
    id_column_name: str = "transect_id",
):
    new_df = pd.DataFrame()
    unique_ids = df[id_column_name].unique()
    for uid in unique_ids:
        new_filename = f"{uid}_{filename}"
        new_df = df[df[id_column_name] == uid]
        new_df.to_csv(os.path.join(save_location, new_filename))

# 1. Enter the path to the csv file that contains the tide corrected time series
# - replace the path below with the path to the csv file
input_file = r"C:\development\doodleverse\coastseg\CoastSeg\sessions\rym1_model_extract_shorelines\transect_time_series_tidally_corrected.csv"
# 2. The output file will be saved in the same directory as the input file
save_location = os.path.dirname(os.path.abspath(input_file))


if not os.path.exists(input_file):
    print(f"File not found: {input_file}")
    exit()

# read the csv file
timeseries_df = pd.read_csv(input_file)
if "Unnamed: 0" in timeseries_df.columns:
    timeseries_df.drop(columns=["Unnamed: 0"], inplace=True)

# save a csv for each transect that was tidally corrected
save_csv_per_id(
    timeseries_df,
    save_location,
    filename="timeseries_tidally_corrected.csv",
)
print(f"Saved the tidally corrected time series for each transect with intersections with the shoreline to {save_location} ")