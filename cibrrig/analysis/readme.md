# Analysis Module

## Overview

The `analysis` module provides tools for single-cell and population-level analyses of neural data. It includes functionality for processing spike data, performing dimensionality reduction, and generating various plots to visualize neural activity.

## Features

- **Single-Cell Analysis**: Tools for analyzing individual neuron activity. Currently creates respiratory modulated phase curves.
- **Population Analysis**: Tools for analyzing and visualizing population-level neural activity.
- **Dimensionality Reduction**: Perform PCA and other dimensionality reduction techniques on neural data.
- **Visualization**: Generate various plots, including raster plots, PCA projections, and event-triggered averages.

## Classes

### `Population`

A class for analyzing and visualizing population-level neural activity.

#### Attributes

- `spike_times` (array-like): Array of spike times.
- `spike_clusters` (array-like): Array of cluster IDs corresponding to each spike time.
- `ndims` (int): Number of dimensions for PCA projection.
- `binsize` (float): Time bin size in seconds.
- `sigma` (float): Standard deviation for Gaussian smoothing in seconds.
- `t0` (float): Start time for analysis.
- `tf` (float): End time for analysis.
- `raster` (np.ndarray): Rasterized spike data.
- `raster_smoothed` (np.ndarray): Smoothed raster data.
- `cbins` (np.ndarray): Array of unique cluster IDs.
- `tbins` (np.ndarray): Array of time bin edges.
- `projection` (np.ndarray): PCA projection of the data.
- `pca` (sklearn.decomposition.PCA): Fitted PCA object.
- `transform` (str): Transformation applied to the raster.
- `projection_speed` (np.ndarray): Speed of movement through PCA space.
- `has_ssm` (bool): Whether a state-space model has been loaded.

## Basic Usage

### Initialization

To initialize a `Population` object, you need to provide spike times and cluster IDs:

```python
from cibrrig.analysis.population import Population

spike_times = ...
spike_clusters = ...
pop = Population(spike_times, spike_clusters,t0=100,tf=250,binsize=0.005,sigma=0.01)
pop.compute_projection()
pop.plot_projection(dims=[0,1])