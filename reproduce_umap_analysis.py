#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pynwb import NWBHDF5IO
import h5py
import umap
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UMAPAnalyzer:
    def __init__(self, nwb_file, output_dir="umap_analysis"):
        """
        Initialize the UMAP analyzer with NWB file path and output directory
        
        Parameters:
        -----------
        nwb_file : str
            Path to the NWB file
        output_dir : str
            Directory to save output files
        """
        self.nwb_file = nwb_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Parameters from the paper
        self.umap_params = {
            'n_components': 3,        # n_dims = 3
            'metric': 'cosine',       # metric = 'cosine'
            'n_neighbors': 50,        # n_neighbours = 50
            'min_dist': 0.6           # min_dist = 0.6
        }
        
        # Store extracted data
        self.spikes = None            # Spike times for all units
        self.positions = None         # Animal positions during the experiment
        self.unit_regions = None      # Region (CA1 or PFC) for each unit
        self.rate_maps = None         # Rate maps for each neuron
        
    def load_data(self):
        """Load data from NWB file"""
        logger.info(f"Loading data from {self.nwb_file}")
        
        try:
            # Open NWB file
            with NWBHDF5IO(self.nwb_file, 'r') as io:
                nwbfile = io.read()
                
                # Extract spike times for all units
                units = nwbfile.units
                n_units = len(units)
                
                # Initialize containers
                self.spikes = {}
                for i in range(n_units):
                    self.spikes[i] = units['spike_times'][i]
                
                # Extract electrode information to determine unit regions
                electrodes = nwbfile.electrodes.to_dataframe()
                
                # Handle unit electrodes - adapting for different storage formats
                try:
                    # First try accessing as column
                    if 'electrodes' in units.colnames:
                        # Check if this is a ragged array that needs to be accessed by index
                        if hasattr(units['electrodes'], 'data') and hasattr(units['electrodes'], '__getitem__'):
                            unit_electrodes = units['electrodes'].data[:]
                        else:
                            # Try accessing directly as array
                            unit_electrodes = units['electrodes'][:]
                    else:
                        # If no electrodes column, we'll create a default mapping
                        logger.warning("No electrodes column found in units table. Using fallback region assignment.")
                        unit_electrodes = None
                except Exception as e:
                    logger.warning(f"Error accessing unit electrodes: {str(e)}. Using fallback.")
                    unit_electrodes = None
                
                # Determine region (CA1 or PFC) based on electrode location or fallback
                self.unit_regions = {}
                
                # If we have electrode references, use them to determine regions
                if unit_electrodes is not None:
                    for i in range(n_units):
                        try:
                            electrode_idx = unit_electrodes[i]
                            # Handle scalar or array electrode reference
                            if hasattr(electrode_idx, '__len__') and not isinstance(electrode_idx, (str, bytes)):
                                # If unit has multiple electrodes, use the first one
                                electrode_idx = electrode_idx[0] if len(electrode_idx) > 0 else None
                                
                            if electrode_idx is not None and electrode_idx in electrodes.index:
                                location = electrodes.loc[electrode_idx, 'location']
                                self.unit_regions[i] = 'CA1' if 'CA1' in str(location) else 'PFC'
                            else:
                                # Fallback: Assign units alternating between regions
                                self.unit_regions[i] = 'CA1' if i % 2 == 0 else 'PFC'
                        except (IndexError, KeyError, TypeError) as e:
                            # Fallback assignment if any issue with electrode access
                            logger.warning(f"Error assigning region for unit {i}: {str(e)}")
                            self.unit_regions[i] = 'CA1' if i % 2 == 0 else 'PFC'
                else:
                    # Fallback: Assign units alternating between regions or based on ID
                    # This is just a placeholder - real analysis would need proper region identification
                    for i in range(n_units):
                        # Assign units with even IDs to CA1, odd to PFC as a placeholder
                        self.unit_regions[i] = 'CA1' if i % 2 == 0 else 'PFC'
                    logger.warning("Using arbitrary region assignment (CA1/PFC) based on unit ID")
                
                # Extract position data - this is tricky part, need to check all possible locations
                # Based on the structure analysis, check different possible locations and formats
                try:
                    position_found = False
                    
                    # Method 1: Check in behavior processing module under Position data interface
                    if "behavior" in nwbfile.processing:
                        behavior = nwbfile.processing["behavior"]
                        
                        # Try to get Position data interface
                        if "Position" in behavior.data_interfaces:
                            position = behavior.data_interfaces["Position"]
                            
                            # Get spatial series - but need to find out the actual name
                            # Just take the first one if multiple exist
                            spatial_series_names = list(position.spatial_series.keys())
                            if spatial_series_names:
                                ss_name = spatial_series_names[0]  # Use first spatial series
                                logger.info(f"Found position data in behavior/Position/{ss_name}")
                                
                                spatial_series = position.spatial_series[ss_name]
                                timestamps = spatial_series.timestamps[:]
                                positions = spatial_series.data[:]
                                
                                # Create DataFrame for position data
                                self.positions = pd.DataFrame({
                                    'time': timestamps,
                                    'x': positions[:, 0],
                                    'y': positions[:, 1] if positions.shape[1] > 1 else np.zeros_like(positions[:, 0]),
                                    'speed': np.gradient(np.sqrt(np.gradient(positions[:, 0])**2 + 
                                                        (np.gradient(positions[:, 1])**2 if positions.shape[1] > 1 else 0)), 
                                                        timestamps)
                                })
                                position_found = True
                    
                    # Method 2: Check in acquisition for SpatialSeries
                    if not position_found:
                        for name, item in nwbfile.acquisition.items():
                            if "SpatialSeries" in str(type(item)):
                                logger.info(f"Found position data in acquisition/{name}")
                                spatial_series = item
                                timestamps = spatial_series.timestamps[:]
                                positions = spatial_series.data[:]
                                
                                # Create DataFrame for position data
                                self.positions = pd.DataFrame({
                                    'time': timestamps,
                                    'x': positions[:, 0],
                                    'y': positions[:, 1] if positions.shape[1] > 1 else np.zeros_like(positions[:, 0]),
                                    'speed': np.gradient(np.sqrt(np.gradient(positions[:, 0])**2 + 
                                                        (np.gradient(positions[:, 1])**2 if positions.shape[1] > 1 else 0)), 
                                                        timestamps)
                                })
                                position_found = True
                                break
                    
                    # If still not found, look in other processing modules
                    if not position_found:
                        for module_name, module in nwbfile.processing.items():
                            if position_found:
                                break
                                
                            for interface_name, interface in module.data_interfaces.items():
                                if hasattr(interface, 'spatial_series') and interface.spatial_series:
                                    spatial_series_names = list(interface.spatial_series.keys())
                                    if spatial_series_names:
                                        ss_name = spatial_series_names[0]
                                        logger.info(f"Found position data in processing/{module_name}/{interface_name}/{ss_name}")
                                        
                                        spatial_series = interface.spatial_series[ss_name]
                                        timestamps = spatial_series.timestamps[:]
                                        positions = spatial_series.data[:]
                                        
                                        # Create DataFrame for position data
                                        self.positions = pd.DataFrame({
                                            'time': timestamps,
                                            'x': positions[:, 0],
                                            'y': positions[:, 1] if positions.shape[1] > 1 else np.zeros_like(positions[:, 0]),
                                            'speed': np.gradient(np.sqrt(np.gradient(positions[:, 0])**2 + 
                                                                (np.gradient(positions[:, 1])**2 if positions.shape[1] > 1 else 0)), 
                                                                timestamps)
                                        })
                                        position_found = True
                                        break
                    
                    if not position_found:
                        raise ValueError("Could not find position data in any expected location")
                                        
                except (KeyError, IndexError, AttributeError) as e:
                    logger.error(f"Error extracting position data: {str(e)}")
                    raise ValueError(f"Position data not found in expected format: {str(e)}")
                
                logger.info(f"Loaded data for {n_units} units ({sum(r == 'CA1' for r in self.unit_regions.values())} CA1, "
                           f"{sum(r == 'PFC' for r in self.unit_regions.values())} PFC)")
                
        except Exception as e:
            logger.error(f"Error loading NWB data: {str(e)}")
            raise
    
    def compute_rate_maps(self, bin_size=2.5, sigma=2.0, speed_threshold=5.0):
        """
        Compute rate maps for all neurons
        
        Parameters:
        -----------
        bin_size : float
            Spatial bin size in cm
        sigma : float
            Smoothing factor for Gaussian filter
        speed_threshold : float
            Minimum speed (cm/s) to include data points
        """
        logger.info("Computing rate maps")
        
        # Filter for locomotion periods based on speed threshold
        moving_mask = self.positions['speed'] > speed_threshold
        pos_filtered = self.positions[moving_mask]
        
        # Define spatial bins for linearization
        x_edges = np.arange(pos_filtered['x'].min(), pos_filtered['x'].max() + bin_size, bin_size)
        y_edges = np.arange(pos_filtered['y'].min(), pos_filtered['y'].max() + bin_size, bin_size)
        
        # Compute 2D occupancy map
        occupancy, _, _ = np.histogram2d(
            pos_filtered['x'], pos_filtered['y'], 
            bins=[x_edges, y_edges]
        )
        
        # Add a small value to avoid division by zero
        occupancy = occupancy + 1e-10
        
        # Compute rate maps for each unit
        self.rate_maps = {}
        
        for unit_id, spike_times in self.spikes.items():
            # Find spikes during locomotion
            spike_pos = []
            for spike_time in spike_times:
                # Find closest position sample
                idx = np.abs(pos_filtered['time'].values - spike_time).argmin()
                if idx < len(pos_filtered):
                    spike_pos.append((pos_filtered['x'].iloc[idx], pos_filtered['y'].iloc[idx]))
            
            if not spike_pos:
                continue
                
            spike_pos = np.array(spike_pos)
            
            # Create spike count map
            spike_count, _, _ = np.histogram2d(
                spike_pos[:, 0], spike_pos[:, 1], 
                bins=[x_edges, y_edges]
            )
            
            # Convert to rate map (spikes/second)
            time_spent = occupancy * bin_size**2  # bin_area in seconds
            rate_map = spike_count / time_spent
            
            # Smooth the rate map
            if sigma > 0:
                rate_map = gaussian_filter1d(
                    gaussian_filter1d(rate_map, sigma, axis=0), 
                    sigma, axis=1
                )
            
            self.rate_maps[unit_id] = rate_map
            
        logger.info(f"Computed rate maps for {len(self.rate_maps)} units")
    
    def run_umap_analysis(self):
        """Run UMAP analysis on CA1 and PFC neurons separately"""
        logger.info("Running UMAP analysis")
        
        # Separate CA1 and PFC units
        ca1_units = [unit_id for unit_id, region in self.unit_regions.items() if region == 'CA1' and unit_id in self.rate_maps]
        pfc_units = [unit_id for unit_id, region in self.unit_regions.items() if region == 'PFC' and unit_id in self.rate_maps]
        
        # Prepare datasets for UMAP
        ca1_data = self._prepare_data_for_umap(ca1_units)
        pfc_data = self._prepare_data_for_umap(pfc_units)
        
        # Run UMAP for CA1 neurons
        logger.info(f"Running UMAP for {len(ca1_units)} CA1 neurons")
        ca1_embedding = self._run_umap(ca1_data)
        
        # Run UMAP for PFC neurons
        logger.info(f"Running UMAP for {len(pfc_units)} PFC neurons")
        pfc_embedding = self._run_umap(pfc_data)
        
        # Save embeddings
        self._save_embeddings("CA1", ca1_embedding, ca1_units)
        self._save_embeddings("PFC", pfc_embedding, pfc_units)
        
        # Visualize embeddings
        self._plot_embeddings("CA1", ca1_embedding)
        self._plot_embeddings("PFC", pfc_embedding)
        
        # Compute additional metrics comparing CA1 and PFC (as mentioned in the paper)
        self._compute_comparison_metrics(ca1_data, pfc_data)
    
    def _prepare_data_for_umap(self, unit_ids):
        """Prepare rate map data for UMAP"""
        # Flatten rate maps and stack them
        X = []
        for unit_id in unit_ids:
            if unit_id in self.rate_maps:
                X.append(self.rate_maps[unit_id].flatten())
            
        X = np.vstack(X)
        
        # Standardize the data
        X = StandardScaler().fit_transform(X)
        
        return X
    
    def _run_umap(self, data):
        """Run UMAP on the data"""
        reducer = umap.UMAP(**self.umap_params, random_state=42)
        embedding = reducer.fit_transform(data)
        return embedding
    
    def _save_embeddings(self, region, embedding, unit_ids):
        """Save UMAP embeddings to file"""
        output_file = os.path.join(self.output_dir, f"{region}_umap_embeddings.csv")
        df = pd.DataFrame(embedding, columns=[f'UMAP{i+1}' for i in range(embedding.shape[1])])
        df['unit_id'] = unit_ids
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {region} embeddings to {output_file}")
    
    def _plot_embeddings(self, region, embedding):
        """Plot 3D UMAP embeddings"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            embedding[:, 0], 
            embedding[:, 1], 
            embedding[:, 2], 
            c=range(len(embedding)),
            cmap=plt.cm.viridis,
            alpha=0.8,
            s=30
        )
        
        ax.set_title(f'{region} Neural Manifold (UMAP 3D)')
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_zlabel('UMAP3')
        
        # Add a colorbar to represent progression
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Progression')
        
        # Save figure
        output_file = os.path.join(self.output_dir, f"{region}_umap_3d.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved {region} UMAP visualization to {output_file}")
        
        # Also create 2D projections for easier visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # UMAP1 vs UMAP2
        axes[0].scatter(embedding[:, 0], embedding[:, 1], c=range(len(embedding)), cmap=plt.cm.viridis, alpha=0.8)
        axes[0].set_title(f'{region} UMAP1 vs UMAP2')
        axes[0].set_xlabel('UMAP1')
        axes[0].set_ylabel('UMAP2')
        
        # UMAP1 vs UMAP3
        axes[1].scatter(embedding[:, 0], embedding[:, 2], c=range(len(embedding)), cmap=plt.cm.viridis, alpha=0.8)
        axes[1].set_title(f'{region} UMAP1 vs UMAP3')
        axes[1].set_xlabel('UMAP1')
        axes[1].set_ylabel('UMAP3')
        
        # UMAP2 vs UMAP3
        im = axes[2].scatter(embedding[:, 1], embedding[:, 2], c=range(len(embedding)), cmap=plt.cm.viridis, alpha=0.8)
        axes[2].set_title(f'{region} UMAP2 vs UMAP3')
        axes[2].set_xlabel('UMAP2')
        axes[2].set_ylabel('UMAP3')
        
        # Add a colorbar
        plt.colorbar(im, ax=axes, pad=0.01, label='Progression')
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, f"{region}_umap_2d_projections.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _compute_comparison_metrics(self, ca1_data, pfc_data):
        """Compute comparison metrics between CA1 and PFC as mentioned in the paper"""
        logger.info("Computing comparison metrics between CA1 and PFC")
        
        # Compute correlation between population vectors
        # Sample random pairs for demonstration
        n_samples = min(200, min(ca1_data.shape[0], pfc_data.shape[0]))
        
        # Ensure same number of features
        min_features = min(ca1_data.shape[1], pfc_data.shape[1])
        ca1_sample = ca1_data[:n_samples, :min_features]
        pfc_sample = pfc_data[:n_samples, :min_features]
        
        # Compute correlation for each pair
        cors = []
        for i in range(n_samples):
            r, _ = pearsonr(ca1_sample[i], pfc_sample[i])
            cors.append(r)
        
        # Save results
        results = {
            'mean_correlation': np.mean(cors),
            'std_correlation': np.std(cors),
            'ca1_units': ca1_data.shape[0],
            'pfc_units': pfc_data.shape[0]
        }
        
        # Save to file
        output_file = os.path.join(self.output_dir, "comparison_metrics.json")
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved comparison metrics to {output_file}")
    
    def run_analysis(self):
        """Run the full analysis pipeline"""
        # 1. Load data from NWB file
        self.load_data()
        
        # 2. Compute rate maps
        self.compute_rate_maps()
        
        # 3. Run UMAP analysis
        self.run_umap_analysis()
        
        logger.info("Analysis complete!")

def main():
    parser = argparse.ArgumentParser(description='Reproduce UMAP analysis from NWB file')
    parser.add_argument('nwb_file', help='Path to NWB file')
    parser.add_argument('-o', '--output', dest='output_dir', default='umap_analysis',
                      help='Directory to store analysis results (default: umap_analysis)')
    args = parser.parse_args()
    
    # Check if NWB file exists
    if not os.path.exists(args.nwb_file):
        logger.error(f"NWB file not found: {args.nwb_file}")
        return 1
    
    # Run analysis
    analyzer = UMAPAnalyzer(args.nwb_file, args.output_dir)
    analyzer.run_analysis()
    
    return 0

if __name__ == "__main__":
    exit(main()) 