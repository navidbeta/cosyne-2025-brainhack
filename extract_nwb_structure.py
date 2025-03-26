#!/usr/bin/env python3
import os
import sys
import json
import h5py
import argparse
import numpy as np
import logging
from pynwb import NWBHDF5IO
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NWBStructureExtractor:
    def __init__(self, nwb_file, output_dir="nwb_structure"):
        """Initialize the NWB structure extractor"""
        self.nwb_file = nwb_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.structure = {}
        
    def extract_using_pynwb(self):
        """Extract structure using the PyNWB library for high-level access"""
        logger.info(f"Extracting high-level structure using PyNWB from {self.nwb_file}")
        try:
            with NWBHDF5IO(self.nwb_file, 'r') as io:
                nwbfile = io.read()
                
                structure = {
                    "file_info": {
                        "path": self.nwb_file,
                        "size_bytes": os.path.getsize(self.nwb_file),
                        "size_mb": os.path.getsize(self.nwb_file) / (1024 * 1024)
                    },
                    "metadata": {
                        "session_description": nwbfile.session_description,
                        "identifier": nwbfile.identifier,
                        "session_start_time": str(nwbfile.session_start_time),
                        "file_create_date": [str(d) for d in nwbfile.file_create_date],
                        "experimenter": nwbfile.experimenter,
                        "experiment_description": nwbfile.experiment_description,
                        "session_id": nwbfile.session_id,
                        "institution": nwbfile.institution,
                        "lab": nwbfile.lab,
                        "keywords": nwbfile.keywords
                    },
                    "acquisition": self._extract_containers(nwbfile.acquisition),
                    "stimulus": self._extract_containers(nwbfile.stimulus),
                    "intervals": self._extract_containers(nwbfile.intervals),
                    "processing": {},
                    "devices": list(nwbfile.devices.keys())
                }
                
                # Extract processing modules
                for module_name, module in nwbfile.processing.items():
                    structure["processing"][module_name] = {
                        "data_interfaces": {}
                    }
                    for interface_name, interface in module.data_interfaces.items():
                        interface_info = self._extract_interface(interface)
                        structure["processing"][module_name]["data_interfaces"][interface_name] = interface_info
                
                # Extract units information if available
                if hasattr(nwbfile, "units") and nwbfile.units is not None:
                    # Get column names
                    cols = list(nwbfile.units.colnames)
                    structure["units"] = {
                        "column_names": cols,
                        "count": len(nwbfile.units),
                        "column_details": {}
                    }
                    
                    # Sample a few values from each column (including spike_times) to understand format
                    for col in cols:
                        try:
                            if col == 'spike_times':
                                # For spike_times, get details but not full data
                                sample_sizes = [len(nwbfile.units[col][i]) for i in range(min(5, len(nwbfile.units)))]
                                structure["units"]["column_details"][col] = {
                                    "type": "ragged array",
                                    "data_type": str(type(nwbfile.units[col][0])),
                                    "sample_sizes": sample_sizes
                                }
                            else:
                                # For other columns, get a sample of values
                                sample = [nwbfile.units[col][i] for i in range(min(5, len(nwbfile.units)))]
                                structure["units"]["column_details"][col] = {
                                    "data_type": str(type(sample[0])) if sample else "unknown",
                                    "sample": str(sample)[:200] + "..." if len(str(sample)) > 200 else str(sample)
                                }
                        except Exception as e:
                            structure["units"]["column_details"][col] = {
                                "error": f"Error sampling: {str(e)}"
                            }
                
                # Extract electrodes information if available
                if hasattr(nwbfile, "electrodes") and nwbfile.electrodes is not None:
                    structure["electrodes"] = {
                        "column_names": list(nwbfile.electrodes.colnames),
                        "count": len(nwbfile.electrodes),
                        "column_details": {}
                    }
                    
                    # Get details about each column
                    for col in nwbfile.electrodes.colnames:
                        try:
                            # Sample a few values
                            sample = [nwbfile.electrodes[col][i] for i in range(min(5, len(nwbfile.electrodes)))]
                            structure["electrodes"]["column_details"][col] = {
                                "data_type": str(type(sample[0])) if sample else "unknown",
                                "sample": str(sample)[:200] + "..." if len(str(sample)) > 200 else str(sample)
                            }
                        except Exception as e:
                            structure["electrodes"]["column_details"][col] = {
                                "error": f"Error sampling: {str(e)}"
                            }
                
                # Extract spatial series information with explicit details
                structure["spatial_series"] = self._extract_spatial_series(nwbfile)
                
                return structure
                
        except Exception as e:
            logger.error(f"Error extracting structure with PyNWB: {str(e)}")
            return {"error": str(e)}
    
    def _extract_spatial_series(self, nwbfile):
        """Extract detailed information about spatial series"""
        spatial_series_info = {}
        
        try:
            # Look in behavior processing module first
            if "behavior" in nwbfile.processing:
                behavior = nwbfile.processing["behavior"]
                
                # Extract position data if available
                if "Position" in behavior.data_interfaces:
                    position = behavior.data_interfaces["Position"]
                    spatial_series_info["Position"] = {
                        "spatial_series": {}
                    }
                    
                    # Get all spatial series in Position
                    for ss_name, ss in position.spatial_series.items():
                        spatial_series_info["Position"]["spatial_series"][ss_name] = {
                            "name": ss.name,
                            "neurodata_type": ss.neurodata_type,
                            "shape": ss.data.shape,
                            "timestamps_shape": ss.timestamps.shape if hasattr(ss.timestamps, "shape") else None,
                            "description": ss.description,
                            "reference_frame": ss.reference_frame
                        }
            
            # Look in acquisition as well
            for name, series in nwbfile.acquisition.items():
                if "SpatialSeries" in str(type(series)):
                    if "direct_acquisition" not in spatial_series_info:
                        spatial_series_info["direct_acquisition"] = {}
                    
                    spatial_series_info["direct_acquisition"][name] = {
                        "name": series.name,
                        "neurodata_type": series.neurodata_type,
                        "shape": series.data.shape,
                        "timestamps_shape": series.timestamps.shape if hasattr(series.timestamps, "shape") else None,
                        "description": series.description,
                        "reference_frame": series.reference_frame if hasattr(series, "reference_frame") else None
                    }
        
        except Exception as e:
            logger.warning(f"Error extracting spatial series info: {str(e)}")
            spatial_series_info["error"] = str(e)
        
        return spatial_series_info
    
    def _extract_containers(self, container_dict):
        """Extract info from a container dictionary"""
        result = {}
        for name, container in container_dict.items():
            try:
                result[name] = {
                    "type": container.__class__.__name__,
                    "neurodata_type": container.neurodata_type
                }
                
                # Extract additional details for some container types
                if hasattr(container, "data"):
                    result[name]["shape"] = container.data.shape
                    result[name]["dtype"] = str(container.data.dtype)
                
                if hasattr(container, "timestamps") and container.timestamps is not None:
                    result[name]["timestamps_shape"] = container.timestamps.shape
            except Exception as e:
                result[name] = {"error": str(e)}
        
        return result
    
    def _extract_interface(self, interface):
        """Extract info from a data interface"""
        try:
            result = {
                "type": interface.__class__.__name__,
                "neurodata_type": interface.neurodata_type
            }
            
            # Handle special interfaces
            if hasattr(interface, "spatial_series"):
                result["spatial_series"] = {}
                for name, ss in interface.spatial_series.items():
                    result["spatial_series"][name] = {
                        "name": ss.name,
                        "shape": ss.data.shape,
                        "timestamps_shape": ss.timestamps.shape if hasattr(ss.timestamps, "shape") else None,
                        "reference_frame": ss.reference_frame if hasattr(ss, "reference_frame") else None
                    }
            
            elif hasattr(interface, "electrical_series"):
                result["electrical_series"] = {}
                for name, es in interface.electrical_series.items():
                    result["electrical_series"][name] = {
                        "name": es.name,
                        "shape": es.data.shape,
                        "timestamps_shape": es.timestamps.shape if hasattr(es.timestamps, "shape") else None
                    }
            
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def extract_using_h5py(self):
        """Extract detailed structure using h5py for low-level access"""
        logger.info(f"Extracting detailed structure using h5py from {self.nwb_file}")
        
        def get_dataset_info(dataset):
            """Get information about an HDF5 dataset"""
            try:
                shape = dataset.shape
                dtype = str(dataset.dtype)
                chunks = dataset.chunks
                size_bytes = dataset.nbytes
                size_mb = size_bytes / (1024 * 1024)
                
                # For small datasets, get a sample
                if dataset.size < 1000:
                    try:
                        sample = dataset[()]
                        if isinstance(sample, (bytes, np.bytes_)):
                            sample = sample.decode('utf-8', errors='replace')
                        elif isinstance(sample, np.ndarray) and sample.dtype.kind == 'S':
                            sample = str(sample)
                        else:
                            sample = str(sample)
                    except Exception as e:
                        sample = f"<Error reading: {str(e)}>"
                else:
                    sample = f"<Large dataset: {shape} elements>"
                
                return {
                    "shape": shape,
                    "dtype": dtype,
                    "chunks": chunks,
                    "size_bytes": size_bytes,
                    "size_mb": size_mb,
                    "sample": sample[:500] + "..." if isinstance(sample, str) and len(sample) > 500 else sample
                }
            except Exception as e:
                return {"error": str(e)}
        
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
            with h5py.File(self.nwb_file, 'r') as file:
                # Extract the HDF5 structure
                h5_structure = extract_group(file)
                
                # Also get file version from attributes if possible
                version = None
                if 'nwb_version' in file.attrs:
                    version = file.attrs['nwb_version']
                    if isinstance(version, (bytes, np.bytes_)):
                        version = version.decode('utf-8', errors='replace')
                
                result = {
                    "h5py_structure": h5_structure,
                    "nwb_version": version,
                    "file_size_bytes": os.path.getsize(self.nwb_file),
                    "file_size_mb": os.path.getsize(self.nwb_file) / (1024 * 1024)
                }
                
                return result
                
        except Exception as e:
            logger.error(f"Error extracting structure with h5py: {str(e)}")
            return {"error": str(e)}
    
    def extract_nwb_structure(self):
        """Extract NWB file structure using both PyNWB and h5py"""
        # Extract structure using PyNWB for high-level info
        pynwb_structure = self.extract_using_pynwb()
        
        # Extract structure using h5py for detailed low-level info
        h5py_structure = self.extract_using_h5py()
        
        # Combine the results
        self.structure = {
            "pynwb_structure": pynwb_structure,
            "h5py_structure": h5py_structure
        }
        
        # Save the results
        self.save_structure()
        
        # Generate a summary
        self.generate_summary()
        
        return self.structure
    
    def save_structure(self):
        """Save the extracted structure to a JSON file"""
        if not self.structure:
            logger.error("No structure information to save")
            return False
        
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
                if isinstance(obj, tuple):
                    return list(obj)
                return super(NumpyEncoder, self).default(obj)
        
        try:
            output_path = os.path.join(self.output_dir, "nwb_detailed_structure.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.structure, f, indent=2, cls=NumpyEncoder)
                
            logger.info(f"Saved detailed structure to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving structure to JSON: {str(e)}")
            return False
    
    def generate_summary(self):
        """Generate a human-readable summary of the NWB file structure"""
        if not self.structure:
            logger.error("No structure information to generate summary")
            return False
        
        try:
            output_path = os.path.join(self.output_dir, "nwb_structure_summary.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# NWB FILE STRUCTURE SUMMARY\n\n")
                
                # File info
                pynwb_structure = self.structure.get("pynwb_structure", {})
                file_info = pynwb_structure.get("file_info", {})
                
                f.write(f"File: {os.path.basename(self.nwb_file)}\n")
                f.write(f"Size: {file_info.get('size_mb', 0):.2f} MB\n")
                f.write(f"NWB Version: {self.structure.get('h5py_structure', {}).get('nwb_version', 'Unknown')}\n\n")
                
                # Metadata
                metadata = pynwb_structure.get("metadata", {})
                if metadata:
                    f.write("## METADATA\n\n")
                    for key, value in metadata.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")
                
                # Acquisition
                acquisition = pynwb_structure.get("acquisition", {})
                if acquisition:
                    f.write("## ACQUISITION\n\n")
                    for name, info in acquisition.items():
                        shape_str = f"Shape: {info.get('shape', 'N/A')}" if "shape" in info else ""
                        f.write(f"- {name}: Type {info.get('type', 'Unknown')} {shape_str}\n")
                    f.write("\n")
                
                # Processing modules
                processing = pynwb_structure.get("processing", {})
                if processing:
                    f.write("## PROCESSING\n\n")
                    for module_name, module_info in processing.items():
                        f.write(f"- {module_name}:\n")
                        for interface_name, interface_info in module_info.get("data_interfaces", {}).items():
                            f.write(f"  - {interface_name} ({interface_info.get('type', 'Unknown')})\n")
                            
                            # List spatial series if any
                            if "spatial_series" in interface_info:
                                for ss_name, ss_info in interface_info["spatial_series"].items():
                                    f.write(f"    - Spatial Series: {ss_name} - Shape: {ss_info.get('shape', 'N/A')}\n")
                            
                            # List electrical series if any
                            if "electrical_series" in interface_info:
                                for es_name, es_info in interface_info["electrical_series"].items():
                                    f.write(f"    - Electrical Series: {es_name} - Shape: {es_info.get('shape', 'N/A')}\n")
                    f.write("\n")
                
                # Spatial Series - detailed section
                spatial_series = pynwb_structure.get("spatial_series", {})
                if spatial_series:
                    f.write("## SPATIAL SERIES (DETAILED)\n\n")
                    
                    if "Position" in spatial_series:
                        f.write("### Position Data:\n")
                        position_series = spatial_series["Position"].get("spatial_series", {})
                        for name, info in position_series.items():
                            f.write(f"- {name}: Shape {info.get('shape', 'N/A')}\n")
                            f.write(f"  Description: {info.get('description', 'N/A')}\n")
                            f.write(f"  Reference Frame: {info.get('reference_frame', 'N/A')}\n\n")
                    
                    if "direct_acquisition" in spatial_series:
                        f.write("### Direct Acquisition Spatial Series:\n")
                        for name, info in spatial_series["direct_acquisition"].items():
                            f.write(f"- {name}: Shape {info.get('shape', 'N/A')}\n")
                            f.write(f"  Description: {info.get('description', 'N/A')}\n")
                            f.write(f"  Reference Frame: {info.get('reference_frame', 'N/A')}\n\n")
                    
                    if "error" in spatial_series:
                        f.write(f"Error extracting spatial series: {spatial_series['error']}\n\n")
                
                # Units information
                units = pynwb_structure.get("units", {})
                if units:
                    f.write("## UNITS\n\n")
                    f.write(f"Count: {units.get('count', 'Unknown')}\n")
                    f.write(f"Columns: {', '.join(units.get('column_names', []))}\n\n")
                    
                    # Provide details for specific columns
                    f.write("### Unit Column Details:\n")
                    for col, details in units.get("column_details", {}).items():
                        f.write(f"- {col}: {details.get('data_type', 'Unknown type')}\n")
                        if "sample" in details:
                            sample = details["sample"]
                            if len(str(sample)) > 100:
                                sample = str(sample)[:100] + "..."
                            f.write(f"  Sample: {sample}\n")
                        elif "sample_sizes" in details:
                            f.write(f"  Sample sizes: {details['sample_sizes']}\n")
                        if "error" in details:
                            f.write(f"  Error: {details['error']}\n")
                    f.write("\n")
                
                # Electrodes information
                electrodes = pynwb_structure.get("electrodes", {})
                if electrodes:
                    f.write("## ELECTRODES\n\n")
                    f.write(f"Count: {electrodes.get('count', 'Unknown')}\n")
                    f.write(f"Columns: {', '.join(electrodes.get('column_names', []))}\n\n")
                    
                    # Provide details for specific electrode columns
                    f.write("### Electrode Column Details:\n")
                    for col, details in electrodes.get("column_details", {}).items():
                        f.write(f"- {col}: {details.get('data_type', 'Unknown type')}\n")
                        if "sample" in details:
                            sample = details["sample"]
                            if len(str(sample)) > 100:
                                sample = str(sample)[:100] + "..."
                            f.write(f"  Sample: {sample}\n")
                        if "error" in details:
                            f.write(f"  Error: {details['error']}\n")
                    f.write("\n")
                
            logger.info(f"Generated structure summary in {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Extract detailed structure from an NWB file")
    parser.add_argument("nwb_file", help="Path to the NWB file")
    parser.add_argument("-o", "--output", dest="output_dir", default="nwb_structure",
                      help="Directory to store results (default: nwb_structure)")
    args = parser.parse_args()
    
    # Check if NWB file exists
    if not os.path.exists(args.nwb_file):
        logger.error(f"NWB file not found: {args.nwb_file}")
        return 1
    
    # Extract structure
    extractor = NWBStructureExtractor(args.nwb_file, args.output_dir)
    extractor.extract_nwb_structure()
    
    logger.info(f"Structure extraction complete. Results saved to {args.output_dir}/")
    return 0

if __name__ == "__main__":
    exit(main()) 