import os 
import pandas as pd

def create_csv_per_transect(
    timeseries_df: pd.DataFrame,
    save_path: str,
    file_extension: str = "_timeseries_raw.csv",
) -> None:
    """
    Create a CSV file per transect from a given timeseries DataFrame.

    Args:
        timeseries_df (pd.DataFrame): The timeseries DataFrame containing transect data.
        save_path (str): The path to save the CSV files.
        file_extension (str, optional): The file extension for the CSV files. Defaults to "_timeseries_raw.csv".

    Returns:
        None
    """
    for transect_id in timeseries_df.columns:
        if transect_id == "dates":
            continue
        print(f"Processing {transect_id}")
        # 3. Save the csv file per transect
        df = pd.DataFrame(
                    {
                        "dates": timeseries_df["dates"],
                        transect_id: timeseries_df[transect_id],

                    },
                )
        # Save to csv file
        fn = f"{transect_id}{file_extension}"
        file_path = os.path.join(save_path, fn)
        df.to_csv(
            file_path, sep=",", index=False
        )



# 1. Enter the path to the csv file that contains the time series
# - replace the path below with the path to the csv file
input_file = r"C:\development\doodleverse\coastseg\CoastSeg\sessions\rym1_model_extract_shorelines\transect_time_series.csv"
# 2. The output file will be saved in the same directory as the input file
save_location = os.path.dirname(os.path.abspath(input_file))


if not os.path.exists(input_file):
    print(f"File not found: {input_file}")
    exit()

# read the csv file
tide_corrected_timeseries_df = pd.read_csv(input_file)

# save a csv for each transect that was tidally corrected
create_csv_per_transect(
    tide_corrected_timeseries_df,
    save_location,
)
print(f"Saved the time series for each transect with intersections with the shoreline to {save_location}")