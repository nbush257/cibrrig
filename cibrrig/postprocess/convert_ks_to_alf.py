"""
convert_ks_to_alf.py

Convert Kilosort output to ALF format and update spike tables after manual curation in phy.
Inspired by ibllib's populate_cluster_table.py but without datajoint dependency.

This script handles:
- Converting initial kilosort output to ALF format
- Updating ALF tables after manual curation in phy
- Handling cluster merges and label changes
- Maintaining cluster UUIDs for tracking
"""

__all__ = [
    'find_sorting_path',
    'detect_cluster_merges', 
    'load_cluster_uuids',
    'handle_cluster_merges',
    'update_cluster_labels',
    'export_initial_alf',
    'update_alf_from_phy',
    'main'
]

import numpy as np
import pandas as pd
import uuid
from pathlib import Path
import click
import logging
from datetime import datetime
import shutil
import warnings

from one import alf
import one.alf.io as alfio
from spikeinterface.core import SortingAnalyzer
from cibrrig.sorting.export_to_alf import ALFExporter

logging.basicConfig()
_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)


def find_sorting_path(session_path, sorter_name):
    """
    Find the sorting path for a given session and sorter.
    
    Args:
        session_path (Path): Path to the session directory
        sorter_name (str): Name of the sorter (e.g., 'kilosort4')
        
    Returns:
        Path: Path to the sorting directory
    """
    session_path = Path(session_path)
    
    # Look for sorting output in common locations
    possible_paths = [
        session_path / sorter_name,
        session_path / "spike_sorting" / sorter_name,
        session_path / f"sorting_{sorter_name}",
        session_path / "raw_ephys_data" / sorter_name,
    ]
    
    for path in possible_paths:
        if path.exists() and path.is_dir():
            return path
            
    raise FileNotFoundError(f"Could not find sorting directory for {sorter_name} in {session_path}")


def detect_cluster_merges(alf_path):
    """
    Detect cluster merges by comparing spikes.clusters.npy with spikes.templates.npy.
    
    Args:
        alf_path (Path): Path to ALF directory containing spike files
        
    Returns:
        dict: Mapping of merged cluster IDs to original template IDs
    """
    alf_path = Path(alf_path)
    
    cluster_path = alf_path / 'spikes.clusters.npy'
    template_path = alf_path / 'spikes.templates.npy'
    
    if not cluster_path.exists() or not template_path.exists():
        _log.warning("Could not find spike cluster/template files for merge detection")
        return {}
    
    # Load current cluster assignments and original template assignments
    phy_clusters = np.load(cluster_path)
    orig_clusters = np.load(template_path)
    
    # Get unique IDs
    id_phy = np.unique(phy_clusters)
    id_orig = np.unique(orig_clusters)
    
    # Find original clusters that have been merged into new clusters
    merged_idx = np.setdiff1d(id_orig, id_phy)
    
    if len(merged_idx) == 0:
        _log.info("No cluster merges detected")
        return {}
    
    # Create mapping from new cluster ID to list of original cluster IDs
    merge_mapping = {}
    for original_id in merged_idx:
        # Find which new cluster this original cluster was merged into
        spike_indices = np.where(orig_clusters == original_id)[0]
        if len(spike_indices) > 0:
            new_cluster_id = phy_clusters[spike_indices[0]]
            if new_cluster_id not in merge_mapping:
                merge_mapping[new_cluster_id] = []
            merge_mapping[new_cluster_id].append(original_id)
    
    _log.info(f"Detected {len(merge_mapping)} cluster merges")
    for new_id, orig_ids in merge_mapping.items():
        _log.info(f"  Cluster {new_id} <- {orig_ids}")
        
    return merge_mapping


def load_cluster_uuids(alf_path):
    """
    Load or create cluster UUIDs.
    
    Args:
        alf_path (Path): Path to ALF directory
        
    Returns:
        pd.DataFrame: DataFrame with cluster_id and uuid columns
    """
    alf_path = Path(alf_path)
    uuid_path = alf_path / 'clusters.uuids.csv'
    
    if uuid_path.exists():
        uuid_df = pd.read_csv(uuid_path)
        _log.info(f"Loaded existing UUIDs for {len(uuid_df)} clusters")
    else:
        # Create new UUIDs for all clusters
        if (alf_path / 'spikes.clusters.npy').exists():
            clusters = np.load(alf_path / 'spikes.clusters.npy')
            unique_clusters = np.unique(clusters)
            uuid_df = pd.DataFrame({
                'cluster_id': unique_clusters,
                'uuids': [str(uuid.uuid4()) for _ in unique_clusters]
            })
            uuid_df.to_csv(uuid_path, index=False)
            _log.info(f"Created new UUIDs for {len(uuid_df)} clusters")
        else:
            _log.warning("No cluster data found, cannot create UUIDs")
            return pd.DataFrame(columns=['cluster_id', 'uuids'])
    
    return uuid_df


def handle_cluster_merges(alf_path, merge_mapping, uuid_df):
    """
    Handle cluster merges by updating UUIDs and creating merge information.
    
    Args:
        alf_path (Path): Path to ALF directory
        merge_mapping (dict): Mapping of merged clusters
        uuid_df (pd.DataFrame): DataFrame with cluster UUIDs
        
    Returns:
        pd.DataFrame: Updated UUID DataFrame
    """
    if not merge_mapping:
        return uuid_df
    
    alf_path = Path(alf_path)
    
    # Create merge information DataFrame
    merge_info = []
    for new_cluster_id, orig_cluster_ids in merge_mapping.items():
        # Get UUIDs for original clusters
        orig_uuids = []
        for orig_id in orig_cluster_ids:
            uuid_row = uuid_df[uuid_df['cluster_id'] == orig_id]
            if len(uuid_row) > 0:
                orig_uuids.append(uuid_row['uuids'].iloc[0])
        
        if orig_uuids:
            # Create or get UUID for merged cluster
            merged_uuid_row = uuid_df[uuid_df['cluster_id'] == new_cluster_id]
            if len(merged_uuid_row) == 0:
                # Create new UUID for merged cluster
                new_uuid = str(uuid.uuid4())
                new_row = pd.DataFrame({
                    'cluster_id': [new_cluster_id],
                    'uuids': [new_uuid]
                })
                uuid_df = pd.concat([uuid_df, new_row], ignore_index=True)
            else:
                new_uuid = merged_uuid_row['uuids'].iloc[0]
            
            merge_info.append({
                'cluster_idx': new_cluster_id,
                'cluster_uuid': new_uuid,
                'merged_idx': tuple(orig_cluster_ids),
                'merged_uuid': tuple(orig_uuids)
            })
    
    # Save merge information
    if merge_info:
        merge_df = pd.DataFrame(merge_info)
        merge_df.to_csv(alf_path / 'merge_info.csv', index=False)
        _log.info(f"Saved merge information for {len(merge_info)} merged clusters")
    
    # Update and save UUID file
    uuid_df.to_csv(alf_path / 'clusters.uuids.csv', index=False)
    
    return uuid_df


def update_cluster_labels(alf_path, uuid_df):
    """
    Update cluster labels from phy output files.
    
    Args:
        alf_path (Path): Path to ALF directory
        uuid_df (pd.DataFrame): DataFrame with cluster UUIDs
        
    Returns:
        pd.DataFrame: Updated cluster information
    """
    alf_path = Path(alf_path)
    
    # Load cluster group file (required)
    cluster_group_path = alf_path / 'cluster_group.tsv'
    if not cluster_group_path.exists():
        raise FileNotFoundError(f"cluster_group.tsv not found in {alf_path}. "
                               "Please run phy curation first or ensure the file exists.")
    
    cluster_group = pd.read_csv(cluster_group_path, sep='\t')
    _log.info(f"Loaded cluster groups for {len(cluster_group)} clusters")
    
    # Load cluster notes file (optional)
    cluster_notes_path = alf_path / 'cluster_notes.tsv'
    if cluster_notes_path.exists():
        cluster_notes = pd.read_csv(cluster_notes_path, sep='\t')
        cluster_info = pd.merge(cluster_group, cluster_notes, on='cluster_id', how='outer')
        _log.info(f"Loaded cluster notes for {len(cluster_notes)} clusters")
    else:
        cluster_info = cluster_group.copy()
        cluster_info['notes'] = None
        _log.info("No cluster notes file found, proceeding without notes")
    
    # Fill NaN values
    cluster_info = cluster_info.fillna('')
    
    # Add UUIDs to cluster info
    cluster_info = pd.merge(cluster_info, uuid_df, on='cluster_id', how='left')
    
    # Add timestamp
    current_time = datetime.now().replace(microsecond=0)
    cluster_info['label_time'] = current_time
    
    return cluster_info


def export_initial_alf(session_path, sorter_name, **kwargs):
    """
    Export initial kilosort output to ALF format using existing ALFExporter.
    
    Args:
        session_path (Path): Path to session directory
        sorter_name (str): Name of sorter
        **kwargs: Additional arguments for ALFExporter
    """
    session_path = Path(session_path)
    sorting_path = find_sorting_path(session_path, sorter_name)
    alf_path = session_path / "alf"
    
    _log.info(f"Exporting initial ALF format from {sorting_path} to {alf_path}")
    
    # Create ALF directory if it doesn't exist
    alf_path.mkdir(exist_ok=True)
    
    # Load the sorting analyzer
    analyzer_path = sorting_path / "sorting_analyzer"
    if not analyzer_path.exists():
        raise FileNotFoundError(f"Sorting analyzer not found at {analyzer_path}. "
                               "Please run spike sorting first.")
    
    analyzer = SortingAnalyzer.load(analyzer_path)
    
    # Export using existing ALFExporter
    exporter = ALFExporter(analyzer, alf_path, **kwargs)
    exporter.run()
    
    _log.info("Initial ALF export completed")


def update_alf_from_phy(session_path, sorter_name):
    """
    Update ALF format files after manual curation in phy.
    
    Args:
        session_path (Path): Path to session directory
        sorter_name (str): Name of sorter
    """
    session_path = Path(session_path)
    sorting_path = find_sorting_path(session_path, sorter_name)
    alf_path = session_path / "alf"
    
    if not alf_path.exists():
        raise FileNotFoundError(f"ALF directory not found at {alf_path}. "
                               "Please run initial conversion first.")
    
    _log.info(f"Updating ALF format from phy curation in {sorting_path}")
    
    # Copy phy output files to ALF directory
    phy_files = ['cluster_group.tsv', 'cluster_notes.tsv', 'spikes.clusters.npy']
    for phy_file in phy_files:
        src_path = sorting_path / phy_file
        dst_path = alf_path / phy_file
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            _log.info(f"Copied {phy_file} from phy output")
        elif phy_file == 'cluster_group.tsv':
            raise FileNotFoundError(f"Required file {phy_file} not found in {sorting_path}. "
                                   "Please complete phy curation first.")
    
    # Detect cluster merges
    merge_mapping = detect_cluster_merges(alf_path)
    
    # Load or create cluster UUIDs
    uuid_df = load_cluster_uuids(alf_path)
    
    # Handle cluster merges
    uuid_df = handle_cluster_merges(alf_path, merge_mapping, uuid_df)
    
    # Update cluster labels
    cluster_info = update_cluster_labels(alf_path, uuid_df)
    
    # Save updated cluster information
    cluster_info.to_csv(alf_path / 'clusters.labels.csv', index=False)
    
    _log.info("ALF update from phy curation completed")
    _log.info(f"Updated labels for {len(cluster_info)} clusters")


@click.command()
@click.argument('session_path', type=click.Path(exists=True, path_type=Path))
@click.argument('sorter_name', default='kilosort4')
@click.option('--update-only', is_flag=True, help='Only update from existing phy output, skip initial export')
@click.option('--export-only', is_flag=True, help='Only perform initial export, skip phy update')
@click.option('--copy-binary', is_flag=True, help='Copy binary recording data')
def main(session_path, sorter_name, update_only, export_only, copy_binary):
    """
    Convert Kilosort output to ALF format and/or update after phy curation.
    
    SESSION_PATH: Path to the session directory containing spike sorting results
    SORTER_NAME: Name of the sorter directory (default: kilosort4)
    """
    try:
        if update_only:
            update_alf_from_phy(session_path, sorter_name)
        elif export_only:
            export_initial_alf(session_path, sorter_name, copy_binary=copy_binary)
        else:
            # Do both: initial export and update from phy
            try:
                export_initial_alf(session_path, sorter_name, copy_binary=copy_binary)
            except Exception as e:
                _log.warning(f"Initial export failed or skipped: {e}")
            
            try:
                update_alf_from_phy(session_path, sorter_name)
            except Exception as e:
                _log.warning(f"Phy update failed or skipped: {e}")
                _log.info("This is normal if phy curation hasn't been completed yet")
        
        _log.info("Conversion completed successfully")
        
    except Exception as e:
        _log.error(f"Conversion failed: {e}")
        raise


if __name__ == '__main__':
    main()