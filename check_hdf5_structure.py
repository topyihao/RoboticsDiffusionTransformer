import os
import h5py
import fnmatch

def print_hdf5_structure(file_path):
    """Print the structure of an HDF5 file"""
    def print_attrs(name, obj):
        print(f"Path: {name}")
        if isinstance(obj, h5py.Dataset):
            print(f"  Shape: {obj.shape}, Type: {obj.dtype}")
            
    print(f"\nFile: {file_path}")
    print("="*50)
    
    with h5py.File(file_path, 'r') as f:
        # First print the top-level keys
        print("Top-level keys:", list(f.keys()))
        
        # Then print detailed structure
        f.visititems(print_attrs)
    
    print("="*50)

def main():
    # Path to your dataset directory
    dataset_dir = "data/datasets/put_marker_into_box"
    
    # Find all HDF5 files
    file_paths = []
    for root, _, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            file_path = os.path.join(root, filename)
            file_paths.append(file_path)
    
    # Print the structure of the first file we find
    if file_paths:
        print_hdf5_structure(file_paths[0])
    else:
        print("No HDF5 files found!")

if __name__ == "__main__":
    main()