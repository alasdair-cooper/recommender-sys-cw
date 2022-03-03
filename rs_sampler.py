from dataclasses import replace
from os.path import exists
from math import radians
from matplotlib.pyplot import axis
from matplotlib.style import use
from sklearn.metrics.pairwise import haversine_distances

import pandas as pd
import numpy as np

import json_to_csv_converter

LOCATIONS = {
    "Montreal" : (45.5017, -73.5673), 
    "Calgary" : (51.0447, -114.0719), 
    "Toronto" : (43.6532, -79.3832), 
    "Pittsburgh" : (40.4406, -79.9959), 
    "Charlotte" : (35.2271, -80.8431), 
    "Urbana-Champaign" : (40.1106, -88.2073), 
    "Phoenix" : (33.4484, -112.0740), 
    "Las Vegas" : (36.1699, -115.1398), 
    "Madison" : (43.0731, -89.4012), 
    "Cleveland" : (41.4993, -81.6944)
    }
R = 6373.0
KM_TO_MILES = 1 / 1.609344

DAYS_OF_THE_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"] 
ATTRIBUTES = ["RestaurantsTakeOut", "RestaurantsDelivery", "WheelchairAccessible", "DogsAllowed"]

def initialize_data_full(convert: bool) -> pd.DataFrame:
    if convert or not exists("./yelp_dataset.nxvcfk/yelp_academic_dataset_business.csv"):
        column_names = json_to_csv_converter.get_superset_of_column_names_from_file("./yelp_dataset.nxvcfk/yelp_academic_dataset_business.json")
        json_to_csv_converter.read_and_write_file("./yelp_dataset.nxvcfk/yelp_academic_dataset_business.json", "./yelp_dataset.nxvcfk/yelp_academic_dataset_business.csv", column_names)
        df = pd.read_csv("./yelp_dataset.nxvcfk/yelp_academic_dataset_business.csv")    
        df = df.sample(1000, random_state=1)
        df = df[df["is_open"] == 1]
        df = df[[
        "name", 
        "address",
        "city",
        "state",
        "postal_code",
        "stars", 
        "hours.Monday",
        "hours.Tuesday", 
        "hours.Wednesday", 
        "hours.Thursday", 
        "hours.Friday", 
        "hours.Saturday", 
        "hours.Sunday", 
        "categories", 
        "attributes.RestaurantsTakeOut", 
        "attributes.RestaurantsDelivery", 
        "attributes.WheelchairAccessible", 
        "attributes.DogsAllowed"
        ]]
        for idx, row in df.iterrows():
            categories = str(row.categories)
            for category in categories.split(", "):
                # Split category into multiple columns with value 1 if present
                # NaN filled with 0 later
                df.loc[idx, f"category.{category}"] = 1
        df.pop("categories")
        df.fillna(0, inplace=True)
        df.to_csv("./yelp_dataset.nxvcfk/yelp_academic_dataset_business.csv")
    else:
        df = pd.read_csv("./yelp_dataset.nxvcfk/yelp_academic_dataset_business.csv", index_col=0)    
    return df

def initialize_sample(sampleSize: int) -> pd.DataFrame:
    df = initialize_data_full(False)

    df = df.sample(sampleSize, random_state=1)
   
    for attribute in ATTRIBUTES:
        # Convert bool strings to float for vector calc later 
        df[f"attributes.{attribute}"] = df[f"attributes.{attribute}"].replace({"True": 1.0, "False": 0.0, "None": 0.0})
        # Make sure dtype is float
        df[f"attributes.{attribute}"] = df[f"attributes.{attribute}"].astype(float)

    return df

def set_dst(row: pd.Series, pos: tuple):
    distance = calculate_dst((row["latitude"], row["longitude"], pos[0], pos[1]))

def calculate_dst(pos: tuple) -> float:
    pointA = [radians(pos[0]), radians(pos[1])]
    pointB = [radians(pos[2]), radians(pos[3])]
    return haversine_distances([pointA, pointB])[1][0] * R