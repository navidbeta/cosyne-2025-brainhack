#!/usr/bin/env python3
import os
import sys
import json
import argparse
import logging
from pynwb import NWBHDF5IO
import h5py
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NWBFormatExtractor:
    def __init__(self, nwb_path, output_dir="nwb_analysis"):
        """Initialize the NWB format extractor with file path and output directory"""
        self.nwb_path = nwb_path
        self.output_dir = output_dir
        self.ensure_output_dirs()
        self.structure = {}
        
    def ensure_output_dirs(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory created: {self.output_dir}")
        
    def extract_format_using_pynwb(self):
        """Extract file format using PyNWB library (for valid NWB files)"""
        logger.info(f"Attempting to open NWB file using PyNWB: {self.nwb_path}")
        try:
            with NWBHDF5IO(self.nwb_path, 'r') as io:
                nwbfile = io.read()
                
                # Extract basic metadata
                metadata = {
                    "session_description": nwbfile.session_description,
                    "identifier": nwbfile.identifier,
                    "session_start_time": str(nwbfile.session_start_time),
                    "file_create_date": [str(d) for d in nwbfile.file_create_date],
                    "experimenter": nwbfile.experimenter,
                    "experiment_description": nwbfile.experiment_description,
                    "session_id": nwbfile.session_id,
                    "institution": nwbfile.institution,
                    "lab": nwbfile.lab
                }
                
                # Extract acquisition data
                acquisition = {}
                for name, data in nwbfile.acquisition.items():
                    acquisition[name] = {
                        "type": type(data).__name__,
                        "shape": data.data.shape if hasattr(data, "data") else "N/A",
                        "dtype": str(data.data.dtype) if hasattr(data, "data") else "N/A"
                    }
                
                # Extract processing modules
                processing = {}
                for module_name, module in nwbfile.processing.items():
                    processing[module_name] = {
                        "data_interfaces": list(module.data_interfaces.keys())
                    }
                
                # Extract units information if available
                units = {}
                if hasattr(nwbfile, "units") and nwbfile.units is not None:
                    units["column_names"] = list(nwbfile.units.colnames)
                    units["count"] = len(nwbfile.units)
                
                # Extract device information
                devices = list(nwbfile.devices.keys())
                
                # Combine all information
                self.structure = {
                    "metadata": metadata,
                    "acquisition": acquisition,
                    "processing": processing,
                    "units": units,
                    "devices": devices,
                    "file_size_bytes": os.path.getsize(self.nwb_path)
                }
                
                # Try to extract electrodes information if available
                try:
                    if hasattr(nwbfile, "electrodes") and nwbfile.electrodes is not None:
                        self.structure["electrodes"] = {
                            "column_names": list(nwbfile.electrodes.colnames),
                            "count": len(nwbfile.electrodes)
                        }
                except Exception as e:
                    logger.warning(f"Could not extract electrodes info: {str(e)}")
                
                # Try to extract epochs if available
                try:
                    if hasattr(nwbfile, "epochs") and nwbfile.epochs is not None:
                        self.structure["epochs"] = {
                            "column_names": list(nwbfile.epochs.colnames),
                            "count": len(nwbfile.epochs)
                        }
                except Exception as e:
                    logger.warning(f"Could not extract epochs info: {str(e)}")
                
                logger.info("Successfully extracted NWB file format using PyNWB")
                return True
                
        except Exception as e:
            logger.warning(f"Error extracting format using PyNWB: {str(e)}")
            return False
    
    def extract_format_using_h5py(self):
        """Extract file format using h5py (HDF5 library) for any NWB file"""
        logger.info(f"Extracting NWB file format using h5py: {self.nwb_path}")
        
        def get_dataset_info(dataset):
            """Get information about an HDF5 dataset"""
            shape = dataset.shape
            dtype = str(dataset.dtype)
            
            # For very large datasets, get a small sample
            if dataset.size > 0 and np.prod(dataset.shape) < 10000:
                try:
                    sample = dataset[()]
                    if isinstance(sample, (bytes, np.bytes_)):
                        sample = sample.decode('utf-8', errors='replace')
                    elif isinstance(sample, np.ndarray) and sample.dtype.kind == 'S':
                        sample = str(sample)
                    else:
                        sample = str(sample)
                        
                    # Truncate if sample is too large
                    if len(sample) > 1000:
                        sample = sample[:1000] + "... [truncated]"
                except Exception as e:
                    sample = f"<Error reading: {str(e)}>"
            else:
                sample = "<Large dataset, sample not shown>"
            
            return {
                "shape": shape,
                "dtype": dtype,
                "sample": sample
            }
        
        def extract_group(group, path="/"):
            """Recursively extract group structure"""
            result = {
                "type": "group",
                "attributes": dict(group.attrs),
                "items": {}
            }
            
            for name, item in group.items():
                item_path = f"{path}{name}/"
                
                if isinstance(item, h5py.Group):
                    result["items"][name] = extract_group(item, item_path)
                elif isinstance(item, h5py.Dataset):
                    result["items"][name] = {
                        "type": "dataset",
                        "info": get_dataset_info(item),
                        "attributes": dict(item.attrs)
                    }
            
            return result
        
        try:
            with h5py.File(self.nwb_path, 'r') as file:
                # Extract the HDF5 structure
                h5_structure = extract_group(file)
                
                # Also get file version from attributes if possible
                version = None
                if 'nwb_version' in file.attrs:
                    version = file.attrs['nwb_version']
                    if isinstance(version, (bytes, np.bytes_)):
                        version = version.decode('utf-8', errors='replace')
                
                self.structure["hdf5_structure"] = h5_structure
                self.structure["nwb_version"] = version
                self.structure["file_size_bytes"] = os.path.getsize(self.nwb_path)
                
                logger.info("Successfully extracted NWB file format using h5py")
                return True
                
        except Exception as e:
            logger.error(f"Error extracting format using h5py: {str(e)}")
            return False
    
    def save_format_to_json(self):
        """Save the extracted format to a JSON file"""
        if not self.structure:
            logger.error("No structure information to save")
            return False
        
        output_path = os.path.join(self.output_dir, "nwb_format.json")
        
        # Custom JSON encoder to handle NumPy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.bool_):
                    return bool(obj)
                if isinstance(obj, (bytes, np.bytes_)):
                    return obj.decode('utf-8', errors='replace')
                return super(NumpyEncoder, self).default(obj)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.structure, f, indent=2, cls=NumpyEncoder)
                
            logger.info(f"NWB file format saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving format to JSON: {str(e)}")
            return False
    
    def save_format_summary(self):
        """Save a human-readable summary of the NWB file format"""
        if not self.structure:
            logger.error("No structure information to save")
            return False
        
        output_path = os.path.join(self.output_dir, "nwb_format_summary.txt")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# NWB FILE FORMAT SUMMARY\n\n")
                
                # File details
                f.write(f"File: {os.path.basename(self.nwb_path)}\n")
                f.write(f"Size: {self.structure.get('file_size_bytes', 0) / (1024*1024):.2f} MB\n")
                f.write(f"NWB Version: {self.structure.get('nwb_version', 'Unknown')}\n\n")
                
                # Metadata if available
                if 'metadata' in self.structure:
                    f.write("## METADATA\n\n")
                    metadata = self.structure['metadata']
                    for key, value in metadata.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")
                
                # Main data categories
                categories = ['acquisition', 'processing', 'units', 'devices', 'electrodes', 'epochs']
                for category in categories:
                    if category in self.structure and self.structure[category]:
                        f.write(f"## {category.upper()}\n\n")
                        
                        if isinstance(self.structure[category], dict):
                            for key, value in self.structure[category].items():
                                f.write(f"- {key}: {value}\n")
                        elif isinstance(self.structure[category], list):
                            for item in self.structure[category]:
                                f.write(f"- {item}\n")
                        else:
                            f.write(f"{self.structure[category]}\n")
                            
                        f.write("\n")
                
                # If we have HDF5 structure, create a simplified tree view
                if 'hdf5_structure' in self.structure:
                    f.write("## HDF5 STRUCTURE\n\n")
                    self._write_hdf5_tree(f, self.structure['hdf5_structure'], level=0)
                
            logger.info(f"NWB file format summary saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving format summary: {str(e)}")
            return False
    
    def _write_hdf5_tree(self, file, structure, level=0, max_level=5):
        """Write a tree view of the HDF5 structure to the summary file"""
        if level > max_level:
            file.write("  " * level + "... (deeper levels not shown)\n")
            return
            
        indent = "  " * level
        
        if structure['type'] == 'group':
            for name, item in structure.get('items', {}).items():
                file.write(f"{indent}- {name}/\n")
                
                # Write important attributes if any
                attrs = item.get('attributes', {})
                if 'neurodata_type' in attrs:
                    file.write(f"{indent}  [neurodata_type: {attrs['neurodata_type']}]\n")
                
                # Recursively write children
                self._write_hdf5_tree(file, item, level + 1, max_level)
        else:
            # Datasets are handled by the parent group logic
            pass
    
    def extract_nwb_format(self):
        """Extract NWB file format using multiple methods"""
        # Try to extract using PyNWB first
        success_pynwb = self.extract_format_using_pynwb()
        
        # If PyNWB fails or provides limited info, use h5py
        if not success_pynwb:
            self.extract_format_using_h5py()
            
        # Save the results to files
        self.save_format_to_json()
        self.save_format_summary()

def main():
    parser = argparse.ArgumentParser(description="Extract the format and structure of an NWB (Neurodata Without Borders) file")
    parser.add_argument("nwb_path", help="Path to the NWB file to analyze")
    parser.add_argument("-o", "--output", dest="output_dir", default="nwb_analysis", 
                        help="Directory to store analysis results (default: nwb_analysis)")
    args = parser.parse_args()
    
    if not os.path.exists(args.nwb_path):
        logger.error(f"NWB file not found: {args.nwb_path}")
        return 1
    
    extractor = NWBFormatExtractor(args.nwb_path, args.output_dir)
    extractor.extract_nwb_format()
    
    logger.info(f"NWB format extraction complete. Results saved to {args.output_dir}/")
    return 0

if __name__ == "__main__":
    exit(main()) 