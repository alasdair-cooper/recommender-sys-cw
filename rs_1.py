from encodings import utf_8
import json
import pandas as pd

import json_to_csv_converter

def initialize_data_full() -> pd.DataFrame:
    column_names = json_to_csv_converter.get_superset_of_column_names_from_file("./yelp_dataset.nxvcfk/yelp_academic_dataset_business.json")
    json_to_csv_converter.read_and_write_file("./yelp_dataset.nxvcfk/yelp_academic_dataset_business.json", "./yelp_dataset.nxvcfk/yelp_academic_dataset_business.csv", column_names)
    df = pd.read_csv("./yelp_dataset.nxvcfk/yelp_academic_dataset_business.csv")
    return df

def initialize_sample(sampleSize: int) -> pd.DataFrame:
    full_df = initialize_data_full()
    sample_df = full_df.sample(sampleSize)
    return sample_df

print(initialize_sample(200).head())
