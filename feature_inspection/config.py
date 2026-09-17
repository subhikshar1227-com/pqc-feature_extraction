"""
Phase 2: Feature Inspection Configuration

Centralized configuration for actual feature detection, matching,
and quality inspection parameters.
"""

# Edge combination weights for final representation
PREPROCESSING_INTERNAL_EDGE_COMBINATION_WEIGHT = 0.7     # Weight for internal geometry edges
PREPROCESSING_OUTER_BOUNDARY_COMBINATION_WEIGHT = 0.8   # Weight for outer boundary edges

# Noise reduction and texture suppression
PREPROCESSING_NOISE_REDUCTION_KERNEL_SIZE = 3           # Kernel size for noise reduction morphology
PREPROCESSING_MIN_EDGE_STRENGTH = 20                    # Minimum edge strength to retain
PREPROCESSING_TEXTURE_SUPPRESSION_THRESHOLD = 15        # Threshold for texture vs geometry distinction

# Geometry preservation parameters - CRITICAL FOR AVOIDING GEOMETRY LOSS
PREPROCESSING_MASK_MORPHOLOGY_ENABLED = False           # Whether to use aggressive morphology on mask
PREPROCESSING_CONSERVATIVE_MASK_KERNEL_SIZE = 3         # Small kernel size for minimal geometry impact  
PREPROCESSING_CONSERVATIVE_MASK_ITERATIONS = 1          # Minimal iterations to preserve thin structures
PREPROCESSING_CONTOUR_BASED_MASK_REFINEMENT = True      # Use contour-based mask refinement instead of morphology
PREPROCESSING_BACKGROUND_ESTIMATION_ENABLED = True      # Enable background estimation for better segmentation
PREPROCESSING_MULTI_THRESHOLD_FUSION = True             # Combine multiple threshold methods for robust segmentation

# Multi-threshold fusion parameters - CRITICAL for robust segmentation without hardcoded weights
PREPROCESSING_OTSU_EVIDENCE_WEIGHT = 0.4                # Weight for Otsu thresholding evidence
PREPROCESSING_BACKGROUND_EVIDENCE_WEIGHT = 0.3          # Weight for background estimation evidence  
PREPROCESSING_GRADIENT_EVIDENCE_WEIGHT = 0.3            # Weight for gradient-based evidence
PREPROCESSING_EVIDENCE_NORMALIZATION_THRESHOLD = 127    # Threshold for evidence normalization (0-255)
PREPROCESSING_GRADIENT_DILATION_KERNEL_SIZE = 5         # Kernel size for gradient evidence dilation
PREPROCESSING_GRADIENT_DILATION_ITERATIONS = 2          # Iterations for gradient evidence dilation

# Contour-based mask refinement parameters
PREPROCESSING_CONTOUR_REFINEMENT_AREA_FACTOR = 0.1      # Multiplier for min component area in contour refinement

# Silhouette validation parameters - CRITICAL for geometry loss detection
PREPROCESSING_SILHOUETTE_ANALYSIS_KERNEL_SIZE = 7       # Kernel size for external region analysis
PREPROCESSING_SILHOUETTE_ANALYSIS_ITERATIONS = 1        # Iterations for external region dilation  
PREPROCESSING_SILHOUETTE_GRADIENT_PERCENTILE = 75       # Percentile for strong gradient threshold (0-100)
PREPROCESSING_SUSPICIOUS_BOUNDARY_THRESHOLD = 0.3       # Threshold for suspicious boundary fraction
PREPROCESSING_WEAK_BOUNDARY_THRESHOLD = 0.5             # Threshold for weak boundary consistency
PREPROCESSING_EXTERNAL_VS_BOUNDARY_RATIO = 1.5          # Ratio threshold for external vs boundary gradient strength

# Texture suppression parameters - CRITICAL for structure preservation  
PREPROCESSING_TEXTURE_SCALE_EVIDENCE_THRESHOLD = 0.3    # Threshold for scale evidence in texture suppression
PREPROCESSING_TEXTURE_COMPACTNESS_THRESHOLD = 5.0       # Compactness threshold for connectivity evidence
PREPROCESSING_TEXTURE_EVIDENCE_COMBINATION_THRESHOLD = 2.0  # Threshold for combining evidence sources (out of 3)
PREPROCESSING_TEXTURE_CLEANUP_KERNEL_SIZE = 3           # Kernel size for final texture cleanup morphology
PREPROCESSING_TEXTURE_STRONG_GRADIENT_MULTIPLIER = 1.5  # Multiplier for strong gradient threshold

# Final edge representation parameters
PREPROCESSING_FINAL_EDGE_CLEANUP_THRESHOLD = 50         # Threshold for final weak edge removal

# Additional algorithm parameters - CORRECTED to eliminate hardcoded values
PREPROCESSING_BACKGROUND_BORDER_FRACTION = 0.05         # Fraction of image size for border region (was hardcoded 0.05)
PREPROCESSING_BACKGROUND_BORDER_MIN_PIXELS = 5          # Minimum border width in pixels (was hardcoded 5)
PREPROCESSING_ADAPTIVE_THRESHOLD_BLOCK_SIZE = 15        # Block size for adaptive thresholding (was hardcoded 15)
PREPROCESSING_ADAPTIVE_THRESHOLD_C = 5                  # Constant subtracted from mean (was hardcoded 5)
PREPROCESSING_GRADIENT_SOBEL_KERNEL_SIZE = 3            # Sobel kernel size for gradient (was hardcoded 3)
PREPROCESSING_HOLE_AREA_FRACTION_THRESHOLD = 0.1        # Fraction for internal hole detection (was hardcoded 0.1)
PREPROCESSING_SIGNIFICANT_REGION_AREA_FRACTION = 0.1    # Fraction for significant region detection (was hardcoded 0.1)
PREPROCESSING_STRUCTURE_AREA_FRACTION_THRESHOLD = 0.05  # Fraction for structure preservation (was hardcoded 0.05)
PREPROCESSING_CONTOUR_PERIMETER_THRESHOLD = 20          # Minimum perimeter for contour preservation (was hardcoded 20)
PREPROCESSING_CONTOUR_CIRCULARITY_THRESHOLD = 0.3       # Minimum circularity for circular features (was hardcoded 0.3)

# =============================================================================
# Image Processing Configuration
# =============================================================================

# Phase 2 Preprocessing Configuration
PREPROCESSING_MAX_RESOLUTION = 1600         # Maximum long edge dimension (matches quick_test.py MAX_LONG_EDGE)
PREPROCESSING_GAUSSIAN_BLUR_KERNEL = 5      # Gaussian blur kernel size for noise reduction

# Product isolation thresholds  
PREPROCESSING_GRADIENT_THRESHOLD = 12       # Morphological gradient threshold for internal edges
PREPROCESSING_MIN_COMPONENT_AREA_PX = 200   # Minimum component area in pixels
PREPROCESSING_BORDER_MARGIN_FRACTION = 0.01 # Border margin as fraction of image size
PREPROCESSING_BORDER_MARGIN_MIN_PX = 3      # Minimum border margin in pixels

# Canny edge detection for outer boundary
PREPROCESSING_CANNY_OUTER_LOW = 50          # Lower Canny threshold
PREPROCESSING_CANNY_OUTER_HIGH = 150        # Upper Canny threshold

# Morphological operations for mask cleanup
PREPROCESSING_MASK_MORPH_KERNEL_SIZE = 9    # Kernel size for morphological operations
PREPROCESSING_MASK_MORPH_CLOSE_ITERS = 3    # Closing iterations
PREPROCESSING_MASK_MORPH_OPEN_ITERS = 2     # Opening iterations

# CLAHE contrast enhancement
PREPROCESSING_CLAHE_CLIP_LIMIT = 2.0        # CLAHE clip limit
PREPROCESSING_CLAHE_TILE_GRID_SIZE = (8, 8) # CLAHE tile grid size

# Internal edge extraction
PREPROCESSING_INTERNAL_EDGE_MAX_DENSITY = 0.08    # Maximum edge density before threshold raising
PREPROCESSING_GRADIENT_STEP_MULTIPLIER = 1.5      # Multiplier for gradient threshold adjustment
PREPROCESSING_GRADIENT_MAX_RETRIES = 4            # Maximum retries for gradient threshold
PREPROCESSING_GRADIENT_KERNEL_SIZE = 3            # Kernel size for morphological gradient
PREPROCESSING_EDGE_CLOSE_KERNEL_SIZE = 2          # Kernel size for edge closing

# Component scoring weights
PREPROCESSING_COMPONENT_AREA_WEIGHT = 0.95        # Weight for area in component scoring
PREPROCESSING_COMPONENT_CENTRALITY_WEIGHT = 0.05  # Weight for centrality in component scoring

# Image preprocessing (legacy - to be consolidated)
IMAGE_MAX_DIMENSION = 1024               # Maximum width/height for processing
IMAGE_GAUSSIAN_BLUR_KERNEL = 5          # Kernel size for noise reduction
IMAGE_BILATERAL_FILTER_D = 9            # Diameter for bilateral filtering
IMAGE_BILATERAL_FILTER_SIGMA_COLOR = 75  # Color sigma for bilateral filter
IMAGE_BILATERAL_FILTER_SIGMA_SPACE = 75  # Space sigma for bilateral filter

# Edge detection  
EDGE_DETECTION_LOW_THRESHOLD = 50        # Lower Canny threshold
EDGE_DETECTION_HIGH_THRESHOLD = 150      # Upper Canny threshold
EDGE_DETECTION_APERTURE_SIZE = 3         # Sobel aperture size
EDGE_DETECTION_L2_GRADIENT = True        # Use L2 gradient norm

# =============================================================================
# Actual Feature Detection Configuration  
# =============================================================================

# Contour detection
CONTOUR_MIN_AREA = 100                   # Minimum contour area (pixels²)
CONTOUR_MAX_AREA = 50000                 # Maximum contour area (pixels²)
CONTOUR_APPROX_EPSILON_FACTOR = 0.02     # Approximation epsilon as fraction of perimeter

# Circle detection
CIRCLE_HOUGH_DP = 2                      # Inverse ratio of accumulator resolution (increased for more selectivity)
CIRCLE_HOUGH_MIN_DIST_FACTOR = 2.0       # Minimum distance between circle centers as fraction of radius (increased)
CIRCLE_HOUGH_PARAM1 = 100                # Upper Canny threshold for edge detection (increased)
CIRCLE_HOUGH_PARAM2 = 50                 # Accumulator threshold for center detection (increased for selectivity)
CIRCLE_MIN_RADIUS_PIXELS = 15            # Minimum circle radius (pixels) - increased from 10
CIRCLE_MAX_RADIUS_PIXELS = 150           # Maximum circle radius (pixels) - decreased from 200

# Circle validation
CIRCLE_CONTOUR_MATCH_THRESHOLD = 0.8     # Minimum match score for circle-contour correspondence
CIRCLE_ROUNDNESS_THRESHOLD = 0.8         # Minimum roundness for circle classification (increased)
CIRCLE_AREA_RATIO_THRESHOLD = 0.8        # Minimum area ratio (contour area / circle area) (increased)

# Through hole detection
HOLE_MIN_ASPECT_RATIO = 0.8              # Minimum width/height ratio for holes
HOLE_MAX_ASPECT_RATIO = 1.2              # Maximum width/height ratio for holes  
HOLE_MIN_SOLIDITY = 0.8                  # Minimum solidity (contour area / convex hull area)
HOLE_DARKNESS_THRESHOLD = 80             # Maximum average brightness for hole interior (absolute)
HOLE_LOCAL_CONTRAST_THRESHOLD = 20       # Minimum brightness difference (interior vs annulus)
HOLE_ANNULUS_WIDTH_PIXELS = 5            # Width of surrounding annulus for contrast comparison
HOLE_MIN_CONTRAST_RATIO = 0.7            # Minimum ratio (hole_brightness / annulus_brightness)

# Rectangular hole detection
RECT_HOLE_MIN_VERTICES = 4               # Minimum vertices for rectangular approximation
RECT_HOLE_MAX_VERTICES = 6               # Maximum vertices for rectangular approximation
RECT_HOLE_CORNER_ANGLE_TOLERANCE = 15    # Tolerance for right angles (degrees)
RECT_HOLE_MIN_WIDTH_PIXELS = 15          # Minimum rectangular hole width (pixels)
RECT_HOLE_MIN_HEIGHT_PIXELS = 15         # Minimum rectangular hole height (pixels)

# Feature confidence calculation
CONFIDENCE_BASE_SCORE = 0.3              # Base confidence for detected features  
CONFIDENCE_ROUNDNESS_WEIGHT = 0.25       # Weight of roundness in confidence
CONFIDENCE_AREA_RATIO_WEIGHT = 0.20      # Weight of area ratio in confidence
CONFIDENCE_EDGE_STRENGTH_WEIGHT = 0.25   # Weight of edge strength in confidence
CONFIDENCE_CONTOUR_QUALITY_WEIGHT = 0.20 # Weight of contour quality in confidence
CONFIDENCE_LOCAL_CONTRAST_WEIGHT = 0.10  # Weight of local contrast in confidence

# Duplicate removal
DUPLICATE_REMOVAL_DISTANCE_THRESHOLD = 30.0  # Distance threshold for duplicate detection (pixels) - increased
DUPLICATE_REMOVAL_SIZE_TOLERANCE = 0.2       # Relative size tolerance for duplicates - decreased for stricter removal

# =============================================================================
# Coordinate Transformation Configuration
# =============================================================================

# Transform validation
TRANSFORM_VALIDATION_TOLERANCE = 1.0     # Tolerance for transform validation (pixels)
COORDINATE_PRECISION_DECIMALS = 3        # Decimal places for coordinate rounding

# =============================================================================
# Feature Matching Configuration
# =============================================================================

# Geometric matching tolerances
MATCH_CENTER_TOLERANCE_MM = 2.0          # Center position tolerance (mm in DXF space)
MATCH_RADIUS_TOLERANCE_ABSOLUTE_MM = 1.0 # Absolute radius tolerance (mm)
MATCH_RADIUS_TOLERANCE_RELATIVE = 0.15   # Relative radius tolerance (15%)

# Matching algorithm
MATCH_ASSIGNMENT_METHOD = "hungarian"     # Assignment algorithm: "greedy" or "hungarian"
MATCH_MIN_CONFIDENCE = 0.6               # Minimum confidence for valid matches
MATCH_TYPE_COMPATIBILITY_BONUS = 0.2     # Bonus for exact feature type match

# Distance weighting
MATCH_DISTANCE_WEIGHT = 0.5              # Weight of center distance in match score
MATCH_SIZE_WEIGHT = 0.3                  # Weight of size similarity in match score  
MATCH_TYPE_WEIGHT = 0.2                  # Weight of type compatibility in match score

# =============================================================================
# Feature Comparison Configuration
# =============================================================================

# Geometric deviation thresholds
COMPARISON_CENTER_TOLERANCE_MM = 1.5     # Acceptable center deviation (mm)
COMPARISON_RADIUS_TOLERANCE_MM = 0.8     # Acceptable radius deviation (mm)
COMPARISON_RADIUS_TOLERANCE_RELATIVE = 0.1  # Acceptable relative radius deviation (10%)

# Comparison scoring
COMPARISON_EXCELLENT_THRESHOLD = 0.95    # Score threshold for excellent match
COMPARISON_GOOD_THRESHOLD = 0.85         # Score threshold for good match
COMPARISON_ACCEPTABLE_THRESHOLD = 0.7    # Score threshold for acceptable match

# =============================================================================
# Quality Inspection Configuration
# =============================================================================

# Feature scoring weights
INSPECTION_MATCHED_FEATURE_WEIGHT = 1.0     # Weight for successfully matched features
INSPECTION_MISSING_FEATURE_PENALTY = 0.8    # Penalty for each missing expected feature
INSPECTION_EXTRA_FEATURE_PENALTY = 0.3      # Penalty for each unexpected actual feature
INSPECTION_POOR_MATCH_PENALTY = 0.5         # Penalty for poorly matched features

# Inspection decision thresholds
INSPECTION_PASS_THRESHOLD = 0.85         # Minimum score for PASS
INSPECTION_REVIEW_THRESHOLD = 0.7        # Minimum score for REVIEW (vs FAIL)

# Quality metrics
QUALITY_COMPLETENESS_WEIGHT = 0.4        # Weight of feature completeness in quality score
QUALITY_ACCURACY_WEIGHT = 0.4           # Weight of geometric accuracy in quality score  
QUALITY_CONFIDENCE_WEIGHT = 0.2         # Weight of detection confidence in quality score

# Missing feature analysis
MISSING_CRITICAL_FEATURE_PENALTY = 1.0   # Penalty for missing critical features
MISSING_STANDARD_FEATURE_PENALTY = 0.6   # Penalty for missing standard features
MISSING_MINOR_FEATURE_PENALTY = 0.3      # Penalty for missing minor features

# Extra feature analysis  
EXTRA_MAJOR_FEATURE_PENALTY = 0.5       # Penalty for major unexpected features
EXTRA_MINOR_FEATURE_PENALTY = 0.2       # Penalty for minor unexpected features

# =============================================================================
# Feature Classification Thresholds
# =============================================================================

# Feature importance classification by size (mm in DXF space)
FEATURE_CRITICAL_MIN_RADIUS = 10.0      # Minimum radius for critical features
FEATURE_MAJOR_MIN_RADIUS = 5.0          # Minimum radius for major features
FEATURE_MINOR_MAX_RADIUS = 2.0          # Maximum radius for minor features

# Feature type priorities for inspection
FEATURE_TYPE_PRIORITIES = {
    "THROUGH_HOLE": 1.0,                 # Highest priority
    "CIRCLE": 0.8,                       # High priority
    "RECTANGULAR_HOLE": 1.0,             # Highest priority
    "SQUARE_HOLE": 1.0                   # Highest priority
}

# =============================================================================
# Visualization Configuration
# =============================================================================

# Display colors (BGR format for OpenCV)
COLOR_EXPECTED_FEATURE = (0, 255, 0)    # Green for expected features
COLOR_ACTUAL_FEATURE = (255, 0, 0)      # Blue for detected actual features
COLOR_MATCHED_FEATURE = (0, 255, 255)   # Yellow for successfully matched
COLOR_MISSING_FEATURE = (0, 0, 255)     # Red for missing expected features
COLOR_EXTRA_FEATURE = (255, 0, 255)     # Magenta for unexpected actual features

# Display parameters
VISUALIZATION_CIRCLE_THICKNESS = 2      # Line thickness for circles
VISUALIZATION_TEXT_FONT_SCALE = 0.5     # Font scale for labels
VISUALIZATION_TEXT_THICKNESS = 1        # Text line thickness
VISUALIZATION_MARKER_SIZE = 5           # Size of center markers

# =============================================================================
# Debug and Logging Configuration
# =============================================================================

# Debug output control
DEBUG_SAVE_INTERMEDIATE_IMAGES = False   # Save preprocessing steps
DEBUG_SAVE_DETECTION_RESULTS = True     # Save detection visualization
DEBUG_VERBOSE_MATCHING = False          # Detailed matching logs

# Output paths
DEBUG_OUTPUT_SUFFIX = "_debug"          # Suffix for debug files
INSPECTION_OUTPUT_SUFFIX = "_inspection" # Suffix for inspection results

# =============================================================================
# Final Micro-Fix Parameters - Removing Last Hard-Coded Values
# =============================================================================

# Multi-scale texture suppression parameters
PREPROCESSING_TEXTURE_MULTI_SCALE_LEVELS = [1, 2, 3]  # Different blur levels for scale consistency
PREPROCESSING_TEXTURE_FINAL_CLEANUP_ITERATIONS = 1     # Morphological open iterations for noise removal