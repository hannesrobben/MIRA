# Image Processing Library Documentation

## Overview

This library provides a comprehensive framework for processing, analyzing, and visualizing image data, with a focus on holographic microscopy and phase/height data. It enables structure detection, segmentation, alignment, phase retrieval, and visualization through a clean, consistent API.

## Library Structure

```
image_processing/
│
├── core/
│   ├── __init__.py                  # Exposes key classes
│   ├── image.py                     # ImageData class
│   └── structure.py                 # Structure, Segment, Background classes
│
├── processing/
│   ├── __init__.py                  # Exposes main processing functions
│   ├── detection.py                 # Structure detection algorithms
│   ├── segmentation.py              # Structure segmentation
│   ├── alignment.py                 # Image alignment operations
│   └── phase_retrieval.py           # Phase-to-height conversions
│
├── visualization/
│   ├── __init__.py
│   ├── plotter_2d.py                # 2D visualization functions
│   └── plotter_3d.py                # 3D visualization functions
│
├── utils/
│   ├── __init__.py
│   ├── exceptions.py                # Custom exceptions
│   └── helpers.py                   # Utility functions
│
├── __init__.py                      # Import and expose main classes
└── processor.py                     # Main façade class: ImageProcessor
```

## Main Components

### 1. Core Classes

#### 1.1 `ImageData` (`core/image.py`)
- Base class for image representation
- Handles data validation, properties, and basic operations
- Key methods: `from_file()`, `get_profile()`, `flip()`, `resize()`, `pad_to_size()`
- Key properties: `shape`, `x_dim`, `y_dim`, `pixel_size`, `is_quadratic`

#### 1.2 `Structure`, `Segment`, `Background` (`core/structure.py`)
- `Structure`: Represents a detected structure in an image
  - Key methods: `segment()`, `from_mask()`, `center()`
  - Key properties: `mask`, `bounds`, `corner_points`, `area`, `perimeter`
- `Segment`: Represents a portion of a structure
  - Key methods: `extract_with_offset()` 
  - Key properties: `mean_value`, `area`, `inner_data`
- `Background`: Represents the background area of an image
  - Key properties: `mean_value`, `std_value`, `mask`

### 2. Processing Modules

#### 2.1 Structure Detection (`processing/detection.py`)
- `detect_structure()`: Main function for structure detection
- `threshold_detection()`: Uses thresholding for detection
- `intensity_detection()`: Uses intensity data for detection
- `adaptive_detection()`: Uses adaptive thresholding
- `clean_structure_mask()`: Cleans up detected masks
- `detect_edges()`: Detects edges in images
- `detect_corners()`: Detects corners in images

#### 2.2 Structure Segmentation (`processing/segmentation.py`)
- `segment_structure()`: Main function for structure segmentation
- `segment_by_grid()`: Grid-based segmentation
- `segment_by_kmeans()`: Uses k-means clustering
- `segment_by_watershed()`: Uses watershed algorithm
- `create_labeled_mask()`: Creates a labeled mask from segments

#### 2.3 Alignment (`processing/alignment.py`)
- `align_structures()`: Main function for structure alignment
- `align_by_edges()`, `align_by_mask()`, `align_by_features()`: Alignment methods
- `calculate_rotation()`: Calculates rotation between images
- `normalize_pixel_size()`: Normalizes pixel sizes between images
- `pad_to_same_size()`: Ensures images have the same dimensions
- `center_structures()`: Centers structures for alignment

#### 2.4 Phase Retrieval (`processing/phase_retrieval.py`)
- `convert_phase_to_height()`: Converts phase to height data
- `convert_height_to_phase()`: Converts height to phase data
- `unwrap_phase()`: Unwraps phase to remove 2π discontinuities
- `full_phase_retrieval()`: Complete phase retrieval pipeline
- `adjust_background()`: Corrects background in structures

### 3. Visualization Modules

#### 3.1 2D Visualization (`visualization/plotter_2d.py`)
- `plot_image()`: Basic 2D image visualization
- `plot_structure()`: Structure visualization with bounds/corners
- `plot_segments()`: Segment visualization with color coding
- `plot_comparison()`: Side-by-side comparison with difference
- `plot_profiles()`: Horizontal and vertical profile plots
- `visualize_edges()`, `visualize_corners()`: Feature visualization
- `create_masked_overlay()`: Creates overlay visualization

#### 3.2 3D Visualization (`visualization/plotter_3d.py`)
- `plot_surface()`: Basic 3D surface plotting
- `plot_structure_3d()`: 3D visualization of structures
- `plot_height_map()`: Height map with contours
- `create_interactive_plot()`: Interactive 3D visualization
- `export_3d_visualization()`: Multi-angle export for documentation

### 4. Utilities

#### 4.1 Exceptions (`utils/exceptions.py`)
- `DataLoadError`: For file loading issues
- `ValidationError`: For input validation problems
- `ProcessingError`: For general processing failures
- `StructureError`: For structure-related issues
- `SegmentationError`: For segmentation failures
- `AlignmentError`: For alignment issues
- `PhaseRetrievalError`: For phase retrieval problems
- `VisualizationError`: For visualization failures

#### 4.2 Helpers (`utils/helpers.py`)
- `get_logger()`: Creates configured logger instances
- `ensure_numpy_array()`: Safe conversion to numpy arrays
- `normalize_pixel_size()`: Standardizes pixel size format
- `butterworth_filter()`: Creates 2D filters
- `phase_to_height()`, `height_to_phase()`: Conversion functions
- Various geometry helpers: `compute_cross_product()`, `point_in_quadrilateral()`, etc.

### 5. Main Interface

#### 5.1 `ImageProcessor` (`processor.py`)
Provides a unified interface for all operations with method chaining:

**Initialization & Loading:**
- `__init__()`: Initialize with optional data
- `load_phase_data()`: Load phase image data
- `load_intensity_data()`: Load intensity image data
- `from_file()`: Create processor from files

**Structure Operations:**
- `detect_structure()`: Detect structure in phase data
- `segment_structure()`: Segment the detected structure
- `align_with()`: Align with another structure

**Phase/Height Operations:**
- `convert_to_height()`: Convert phase to height
- `convert_to_phase()`: Convert height to phase
- `unwrap_phase()`: Unwrap phase data
- `phase_retrieval()`: Perform full phase retrieval
- `adjust_background()`: Adjust structure background

**Visualization:**
- `plot_image()`: Plot 2D image
- `plot_structure()`: Plot structure with highlights
- `plot_segments()`: Plot segmented structure
- `plot_3d()`: Plot 3D surface
- `plot_height_map()`: Plot height map with contours

**Data Saving:**
- `save_data()`: Save phase or structure data

## Usage Examples

### Basic Usage

```python
# Load data from files
processor = ImageProcessor.from_file("phase.npy", "intensity.npy")

# Detect and segment structure
processor.detect_structure(method='threshold')
processor.segment_structure(method='grid', num_segments=(3, 3))

# Convert to height and visualize
processor.convert_to_height()
processor.plot_3d()
```

### Method Chaining

```python
# Process data in a single chain
ImageProcessor.from_file("phase.npy", "intensity.npy") \
    .detect_structure() \
    .segment_structure() \
    .adjust_background() \
    .convert_to_height() \
    .plot_height_map()
```

### Advanced Processing

```python
# Load and prepare data
processor1 = ImageProcessor.from_file("sample1.npy")
processor2 = ImageProcessor.from_file("sample2.npy")

# Process first sample
processor1.detect_structure() \
    .segment_structure(method='kmeans', num_segments=(2, 2))

# Process second sample
processor2.detect_structure() \
    .segment_structure(method='grid', num_segments=(2, 2))

# Align samples and perform phase retrieval
processor1.align_with(processor2, method='edge') \
    .phase_retrieval(processor2, convert_to_height=True) \
    .plot_comparison()
```

## Key Features

1. **Unified Interface**: Clean, consistent API through the `ImageProcessor` class
2. **Method Chaining**: Fluent API with all processing methods returning `self`
3. **Flexible Input Handling**: Support for different input types (arrays, files, objects)
4. **Comprehensive Error Handling**: Specific exceptions with informative messages
5. **Algorithm Alternatives**: Multiple approaches for each processing task
6. **Visualization Options**: Rich set of 2D and 3D visualization functions
7. **Modular Design**: Well-organized components for maintainability and extensibility

## Dependencies

- **Core**: numpy, scipy
- **Processing**: scikit-image (optional, for advanced algorithms)
- **Visualization**: matplotlib
- **Optional**: scikit-learn (for k-means segmentation)

## Error Handling

All operations use specific exception types for better error handling:

```python
try:
    processor.detect_structure().segment_structure()
except ValidationError as e:
    print(f"Invalid input: {e}")
except ProcessingError as e:
    print(f"Processing failed: {e}")
except SegmentationError as e:
    print(f"Segmentation failed: {e}")
```

## Tips & Best Practices

1. **Use method chaining** for cleaner code
2. **Check if operations succeeded** by validating that data properties exist
3. **Use logging** to debug complex operations
4. **Provide pixel_size** when loading data from arrays
5. **Set appropriate parameters** for detection and segmentation based on your data
6. **Save intermediate results** for complex processing pipelines
7. **Use the appropriate visualization** for your data type and analysis needs