# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 13:27:47 2025

@author: Keerati
"""

import pandas as pd
import os

# Define paths
hf_path = 'D:/Waveforms/Normal/'
seg_path = 'D:/Waveforms/Normal_seg/'

# Ensure the segment folder exists
os.makedirs(seg_path, exist_ok=True)

# List all files in the HF folder
figur_hf = os.listdir(hf_path)

# Columns to select
feature_cols = ['Time', 'Channel A', 'Channel B']

# Loop through each file in the HF folder
for file_name in figur_hf:
    # Construct the full file path
    file_path = os.path.join(hf_path, file_name)
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Select only the desired columns
    df = df[feature_cols]
    
    # Split the data into segments of 1000 rows each
    num_segments = len(df) // 1000  # Determine the number of full segments
    for i in range(num_segments):
        # Extract the segment
        segment = df.iloc[i * 1000:(i + 1) * 1000]
        
        # Define the output file name
        segment_file_name = f"{os.path.splitext(file_name)[0]}_segment_{i + 1}.csv"
        segment_file_path = os.path.join(seg_path, segment_file_name)
        
        # Save the segment to a new CSV file
        segment.to_csv(segment_file_path, index=False)

print("Segmentation completed. All files saved.")