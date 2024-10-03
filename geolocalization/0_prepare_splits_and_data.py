import os
import json
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd
import geopandas as gpd
import re

print("Aligning images")

geojson_path = 'CVIAN/02_Position/CVIAN_position.geojson'
svi_dir = 'CVIAN/00_SVI'
aligned_dir = 'CVIAN/00_SVI_aligned'

os.makedirs(aligned_dir, exist_ok=True)


with open(geojson_path, 'r') as f:
    geojson_data = json.load(f)

# Extract compass angles and associate them with their respective image IDs
image_angles = {}
for feature in geojson_data['features']:
    image_id = feature['properties']['id']
    compass_an = feature['properties']['compass_an']
    image_angles[image_id] = compass_an


# Function to roll the image based on the compass angle
def roll_image(image, angle):
    width = image.width
    roll_pixels = int((angle / 360.0) * width)
    image_array = np.array(image)
    rolled_array = np.roll(image_array, roll_pixels, axis=1)
    rolled_image = Image.fromarray(rolled_array)
    return rolled_image


# Process each image in the svi_dir
for root, _, files in os.walk(svi_dir):
    for file in files:
        if file.endswith('.png'):
            image_id = file.split('.')[0]
            angle = image_angles.get(image_id, None)
            if angle is not None:
                roll_angle = -angle
                original_image_path = os.path.join(root, file)
                image = Image.open(original_image_path)
                rolled_image = roll_image(image, roll_angle)
                relative_path = os.path.relpath(root, svi_dir)
                new_image_dir = os.path.join(aligned_dir, relative_path)
                os.makedirs(new_image_dir, exist_ok=True)  
                new_image_path = os.path.join(new_image_dir, file)
                rolled_image.save(new_image_path)
                print(f"Rolled {file} by {roll_angle} degrees and saved to {new_image_path}")

print("All images have been processed and saved in the aligned directory.")

print("Starting to generate datasplits")

# Paths
svi_dir = 'CVIAN/00_SVI_aligned'
sat_dir = 'CVIAN/01_Satellite'


svi_files = []
sat_files = []

for root, _, files in os.walk(svi_dir):
    for file in files:
        if file.endswith('.png'):
            relative_path = os.path.relpath(os.path.join(root, file), start='CVIAN')
            svi_files.append(relative_path)

for root, _, files in os.walk(sat_dir):
    for file in files:
        if file.endswith('.png'):
            relative_path = os.path.relpath(os.path.join(root, file), start='CVIAN')
            sat_files.append(relative_path)

if not svi_files:
    raise ValueError("No .png files found in 00_SVI directory.")
if not sat_files:
    raise ValueError("No .png files found in 01_Satellite directory.")

svi_files = sorted(svi_files)
sat_files = sorted(sat_files)

paired_files = list(zip(sat_files, svi_files))

df = pd.DataFrame(paired_files, columns=["sat", "ground"])

if df.empty:
    raise ValueError("The dataframe is empty. Check if the files are paired correctly.")


# Function to create train/val splits
def create_splits(df, train_percentage):
    if len(df) == 0:
        raise ValueError("The dataframe is empty, cannot create splits.")
    train_df, val_df = train_test_split(df, train_size=train_percentage / 100, random_state=42)
    return train_df, val_df


# Percentages for training data
percentages = [10, 20, 30, 40, 50, 60]

for pct in percentages:
    try:
        train_df, val_df = create_splits(df, pct)
        train_df.to_csv(f'CVIAN/train_{pct}.csv', index=False, header=None)
        val_df.to_csv(f'CVIAN/val_{pct}.csv', index=False, header=None)
    except ValueError as e:
        print(f"Skipping {pct}% split: {e}")

# Save the 100% evaluation set
df.to_csv('CVIAN/val_100.csv', index=False, header=None)

positions_gdf = gpd.read_file('CVIAN/02_Position/CVIAN_position.geojson')

# Ensure the CRS is set correctly
positions_gdf = positions_gdf.set_crs(epsg=4326)

# Read the subarea files and combine them
subareas = []
for i in range(1, 6):
    subarea_file = f'subarea_{i}.geojson'
    subarea_gdf = gpd.read_file(subarea_file)
    subarea_gdf['subarea'] = i
    subareas.append(subarea_gdf)

# Combine all subareas into one GeoDataFrame
subareas_gdf = pd.concat(subareas, ignore_index=True)
subareas_gdf = subareas_gdf.set_crs(epsg=4326)

# Determine which IDs correspond to which subareas
positions_with_subareas = gpd.sjoin(
    positions_gdf,
    subareas_gdf[['subarea', 'geometry']],
    how='left',
    predicate='within'
)

svi_data = []
sat_data = []


# Function to extract ID from file name
def extract_id(filename):
    match = re.search(r'(\d+)\.png', filename)
    if match:
        return match.group(1)
    else:
        return None


# Collect SVI files
for root, _, files in os.walk(svi_dir):
    for file in files:
        if file.endswith('.png'):
            id = extract_id(file)
            if id:
                relative_path = os.path.relpath(os.path.join(root, file), start='CVIAN')
                svi_data.append({'id': id, 'svi_file': relative_path})

# Collect Satellite files
for root, _, files in os.walk(sat_dir):
    for file in files:
        if file.endswith('.png'):
            id = extract_id(file)
            if id:
                relative_path = os.path.relpath(os.path.join(root, file), start='CVIAN')
                sat_data.append({'id': id, 'sat_file': relative_path})

svi_df = pd.DataFrame(svi_data)
sat_df = pd.DataFrame(sat_data)

# Merge on 'id' to get the file paths together
files_df = pd.merge(sat_df, svi_df, on='id')

# Merge with the positions and subarea assignments
merged_df = pd.merge(positions_with_subareas, files_df, on='id')

# Create train and validation CSV files for each subarea
for subarea in range(1, 6):
    # Validation set: data from the current subarea
    val_df = merged_df[merged_df['subarea'] == subarea][['sat_file', 'svi_file']]
    # Training set: data not from the current subarea
    train_df = merged_df[
        (merged_df['subarea'] != subarea) | (merged_df['subarea'].isna())
    ][['sat_file', 'svi_file']]
    train_df.to_csv(f'CVIAN/train_subarea{subarea}.csv', index=False, header=None)
    val_df.to_csv(f'CVIAN/val_subarea{subarea}.csv', index=False, header=None)

print("CSV files have been generated.")
