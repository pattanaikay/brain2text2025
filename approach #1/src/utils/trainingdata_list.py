"""
Utility script to generate a list of all HDF5 files in the hdf5_data_final directory.

Saves as JSON format for easy structured data handling.
"""

from pathlib import Path
import json
import os


def get_h5_files(data_dir: str) -> list:
    """
    Recursively find all .hdf5 files in the given directory.
    
    Args:
        data_dir (str): Path to the hdf5_data_final directory
        
    Returns:
        list: List of absolute paths to all .hdf5 files, sorted chronologically
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    
    # Recursively find all .hdf5 files
    h5_files = sorted(data_path.glob('**/*.hdf5'))
    
    # Convert to string paths
    h5_list = [str(f) for f in h5_files]
    
    return h5_list


def load_h5_list_from_file(filepath: str = "h5_list_data.json") -> list:
    """
    Load the h5_list from a previously saved JSON file.
    
    Args:
        filepath (str): Path to the saved h5_list JSON file
        
    Returns:
        list: List of h5 file paths
    """
    with open(filepath, "r") as file:
        h5_list = json.load(file)
    return h5_list


# Define the main hdf5_data_final path
# Adjust this path based on your project structure
HDF5_DATA_DIR = r"C:\Projects\Brain2Text2025\t15_copyTask_neuralData\hdf5_data_final"

# Generate the h5_list
h5_list = get_h5_files(HDF5_DATA_DIR)

# Save as JSON file
with open("h5_list_data.json", "w") as file:
    json.dump(h5_list, file, indent=4)

if __name__ == "__main__":
    # Print summary
    print(f"Found {len(h5_list)} .hdf5 files in {HDF5_DATA_DIR}")
    print("\nFirst 5 files:")
    for i, file in enumerate(h5_list[:5]):
        print(f"  {i+1}. {file}")
    print("\nLast 5 files:")
    for i, file in enumerate(h5_list[-5:], start=len(h5_list)-4):
        print(f"  {i}. {file}")
    
    # Verify the saved file
    print(f"\n✓ Saved {len(h5_list)} file paths to 'h5_list_data.json'")
