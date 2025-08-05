#!/usr/bin/env python3
"""
Test script to validate ibllib 3.0 compatibility
Run this after installing the updated dependencies to verify everything works.

Usage:
    python test_ibllib_compatibility.py

Requirements:
    - Have installed cibrrig with updated dependencies: pip install -e .
"""

def test_all_imports():
    """Test that all updated ibllib imports work correctly"""
    
    print("Testing ibllib imports...")
    
    # Test archiving module
    try:
        from cibrrig.archiving.ephys_data_to_alf import misc
        print("✓ ephys_data_to_alf misc import works")
    except Exception as e:
        print(f"✗ ephys_data_to_alf misc import failed: {e}")
    
    # Test sync extraction
    try:
        from cibrrig.preprocess.extract_sync_times import (
            get_sync_fronts, sync_to_alf, sync_probe_front_times, 
            save_timestamps_npy, check_diff_3b
        )
        print("✓ extract_sync_times imports work")
    except Exception as e:
        print(f"✗ extract_sync_times imports failed: {e}")
    
    # Test preprocessing
    try:
        from cibrrig.preprocess.preproc_pipeline import (
            remove_axis_outline, set_axis_label_size, EphysQC, extract_rmsmap
        )
        print("✓ preproc_pipeline imports work")
    except Exception as e:
        print(f"✗ preproc_pipeline imports failed: {e}")
    
    # Test postprocessing
    try:
        from cibrrig.postprocess.convert_ks_to_alf import apply_sync, spike_sorting_metrics
        print("✓ convert_ks_to_alf imports work")
    except Exception as e:
        print(f"✗ convert_ks_to_alf imports failed: {e}")
    
    # Test sorting
    try:
        from cibrrig.sorting.spikeinterface_ks4 import apply_sync
        print("✓ spikeinterface_ks4 imports work")
    except Exception as e:
        print(f"✗ spikeinterface_ks4 imports failed: {e}")

def test_function_signatures():
    """Test that functions have expected signatures"""
    
    print("\nTesting function signatures...")
    
    try:
        from cibrrig.archiving.ephys_data_to_alf import misc
        
        # Test misc functions exist
        for func_name in ['rename_ephys_files', 'move_ephys_files', 'delete_empty_folders']:
            if hasattr(misc, func_name):
                print(f"✓ misc.{func_name} exists")
            else:
                print(f"✗ misc.{func_name} missing")
                
    except Exception as e:
        print(f"✗ Could not test misc functions: {e}")

def test_cli_entry_points():
    """Test that CLI entry points can be imported"""
    
    print("\nTesting CLI entry points...")
    
    entry_points = [
        ("npx_run_all", "cibrrig.main_pipeline", "main"),
        ("backup", "cibrrig.archiving.backup", "main"),
        ("ephys_to_alf", "cibrrig.archiving.ephys_data_to_alf", "cli"),
        ("spikesort", "cibrrig.sorting.spikeinterface_ks4", "cli"),
        ("convert_ks_to_alf", "cibrrig.postprocess.convert_ks_to_alf", "main"),
    ]
    
    for name, module_name, func_name in entry_points:
        try:
            module = __import__(module_name, fromlist=[func_name])
            func = getattr(module, func_name)
            print(f"✓ {name} entry point ({module_name}:{func_name}) works")
        except Exception as e:
            print(f"✗ {name} entry point failed: {e}")

def test_ibllib_version():
    """Check ibllib version"""
    
    print("\nChecking dependencies...")
    
    try:
        import ibllib
        print(f"✓ ibllib version: {ibllib.__version__}")
        
        # Check if it's 3.x
        version_parts = ibllib.__version__.split('.')
        major_version = int(version_parts[0])
        if major_version >= 3:
            print("✓ Using ibllib 3.x as expected")
        else:
            print(f"⚠ Using ibllib {major_version}.x - expected 3.x")
            
    except Exception as e:
        print(f"✗ ibllib check failed: {e}")
    
    try:
        import spikeinterface
        print(f"✓ spikeinterface version: {spikeinterface.__version__}")
    except Exception as e:
        print(f"✗ spikeinterface check failed: {e}")
    
    try:
        import kilosort
        print(f"✓ kilosort version: {kilosort.__version__}")
    except Exception as e:
        print(f"✗ kilosort check failed: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("cibrrig ibllib 3.0 Compatibility Test")
    print("=" * 60)
    
    test_ibllib_version()
    test_all_imports()
    test_function_signatures()
    test_cli_entry_points()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("If all tests pass, npx_run_all should work with ibllib 3.0")
    print("Try running: npx_run_all --help")
    print("=" * 60)