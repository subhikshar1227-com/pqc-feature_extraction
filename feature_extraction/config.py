"""
Feature Extraction Configuration

Centralized configuration for all feature detection thresholds and parameters.
All magic numbers and tunable parameters should be defined here.
"""

from typing import Dict, Any

# =============================================================================
# Geometric Tolerance Configuration
# =============================================================================

# Distance tolerance for considering points/centers as coincident (mm)
GEOMETRIC_TOLERANCE = 0.1

# Angular tolerance for arc continuity (degrees)  
ANGULAR_TOLERANCE = 1.0

# Radius tolerance for considering circles/arcs as similar size
RADIUS_TOLERANCE_ABSOLUTE = 0.5  # mm
RADIUS_TOLERANCE_RELATIVE = 0.05  # 5%

# Concentricity tolerance for nested geometry
CONCENTRICITY_TOLERANCE = 0.2  # mm

# =============================================================================
# Arc-to-Circle Reconstruction Configuration  
# =============================================================================

# Center position tolerance for arc compatibility (mm)
ARC_CENTER_TOLERANCE = 0.2

# Radius tolerance for arc compatibility (mm) 
ARC_RADIUS_TOLERANCE = 0.3

# Angular tolerance for arc continuity (degrees)
ARC_ANGULAR_CONTINUITY_TOLERANCE = 5.0

# Minimum coverage required to reconstruct a circle from arcs (0.0 to 1.0)
CIRCLE_RECONSTRUCTION_MIN_COVERAGE = 0.85  # Require 85% coverage

# Special coverage threshold for semi-circle pairs
CIRCLE_RECONSTRUCTION_SEMICIRCLE_MIN_COVERAGE = 0.95  # 95% for semi-circles

# Maximum gap angle between arcs for circle reconstruction (degrees)
CIRCLE_RECONSTRUCTION_MAX_GAP = 30.0

# Maximum overlap between arcs in reconstruction (degrees)
CIRCLE_RECONSTRUCTION_MAX_OVERLAP = 10.0

# Minimum number of arcs required for reconstruction
CIRCLE_RECONSTRUCTION_MIN_ARCS = 2

# Maximum number of arcs that can be combined into one circle
CIRCLE_RECONSTRUCTION_MAX_ARCS = 8

# Semi-circle detection parameters
SEMICIRCLE_ARC_SPAN_MIN = 170.0  # degrees - minimum span for semi-circle arc
SEMICIRCLE_ARC_SPAN_MAX = 190.0  # degrees - maximum span for semi-circle arc
SEMICIRCLE_ARC_SPAN_TOLERANCE = 15.0  # degrees - tolerance around 180°

# Coverage calculation parameters
COVERAGE_MERGE_TOLERANCE = 1.0    # degrees - tolerance for merging intervals
COVERAGE_WRAPAROUND_THRESHOLD = 350.0  # degrees - threshold for full circle detection
COVERAGE_TINY_GAP_THRESHOLD = 1.0      # degrees - ignore gaps smaller than this

# Reconstruction confidence calculation weights
RECONSTRUCTION_COVERAGE_WEIGHT = 0.4
RECONSTRUCTION_GAP_WEIGHT = 0.3
RECONSTRUCTION_OVERLAP_WEIGHT = 0.2
RECONSTRUCTION_ARC_COUNT_WEIGHT = 0.1

# Minimum confidence threshold for accepting reconstruction
RECONSTRUCTION_MIN_CONFIDENCE = 0.8

# =============================================================================
# Feature Detection Configuration
# =============================================================================

# Circle Detection
# Minimum radius for considering a circle as a prominent feature (mm)
CIRCLE_MIN_RADIUS = 1.5

# Maximum radius for very large structural circles (mm)  
CIRCLE_MAX_RADIUS = 50.0

# Main body detection - circles larger than this are likely structural/main body
MAIN_BODY_MIN_RADIUS = 50.0

# Minimum radius for reconstructed circles to be considered main body
RECONSTRUCTED_MAIN_BODY_MIN_RADIUS = 50.0

# Minimum arc count for large reconstructed circles to be considered main body  
RECONSTRUCTED_MAIN_BODY_MIN_ARC_COUNT = 4

# Feature size ranges for confidence scoring
FEATURE_IDEAL_MIN_RADIUS = 2.0
FEATURE_IDEAL_MAX_RADIUS = 20.0
FEATURE_ACCEPTABLE_MIN_RADIUS = 1.5  
FEATURE_ACCEPTABLE_MAX_RADIUS = 30.0

# Through Hole Detection
# Minimum radius for through hole candidates (mm)
THROUGH_HOLE_MIN_RADIUS = 1.0

# Maximum radius for through hole candidates (mm)
THROUGH_HOLE_MAX_RADIUS = 20.0

# Minimum confidence required for hole classification
HOLE_CLASSIFICATION_MIN_CONFIDENCE = 0.85

# =============================================================================
# Feature Deduplication Configuration
# =============================================================================

# Center tolerance for considering features as duplicates (mm)
FEATURE_DUPLICATE_CENTER_TOLERANCE = 0.5

# Radius tolerance for considering features as duplicates (mm)
FEATURE_DUPLICATE_RADIUS_TOLERANCE = 0.3

# Radius tolerance as relative fraction - CONSERVATIVE to avoid false positives
FEATURE_DUPLICATE_RADIUS_RELATIVE = 0.05  # 5% instead of 10%

# =============================================================================
# Significance Filtering Configuration
# =============================================================================

# Minimum confidence score for including a feature in final results
MIN_FEATURE_CONFIDENCE = 0.65

# Minimum radius for feature significance (mm)
SIGNIFICANCE_MIN_RADIUS = 1.2

# Location-based deduplication tolerance
SIGNIFICANCE_LOCATION_TOLERANCE = 1.0               # mm - tolerance for same location detection

# Hole size scoring values for concentric selection
SIGNIFICANCE_HOLE_SIZE_SCORE_VERY_SMALL = 2         # Score for very small holes (pins, screws)
SIGNIFICANCE_HOLE_SIZE_SCORE_SMALL = 3              # Score for small holes (small bolts)
SIGNIFICANCE_HOLE_SIZE_SCORE_MEDIUM = 4             # Score for medium holes (standard bolts) - PREFERRED
SIGNIFICANCE_HOLE_SIZE_SCORE_LARGE = 3              # Score for large holes (large bolts)
SIGNIFICANCE_HOLE_SIZE_SCORE_VERY_LARGE = 1         # Score for very large holes (less common)

# Circle size scoring values for concentric selection  
SIGNIFICANCE_CIRCLE_SIZE_SCORE_IDEAL = 3            # Score for ideal feature size circles
SIGNIFICANCE_CIRCLE_SIZE_SCORE_ACCEPTABLE = 2       # Score for acceptable feature size circles
SIGNIFICANCE_CIRCLE_SIZE_SCORE_SMALL = 1            # Score for small but possible circles
SIGNIFICANCE_CIRCLE_SIZE_SCORE_POOR = 0             # Score for too large or too small circles

# Confidence weighting factor in selection scoring
SIGNIFICANCE_CONFIDENCE_WEIGHT_FACTOR = 0.5         # Weight factor for confidence in scoring

# Pattern evidence configuration
PATTERN_MIN_SIMILAR_COUNT = 2  # Minimum similar features to constitute a pattern
PATTERN_CENTER_TOLERANCE = 5.0  # mm - tolerance for pattern spacing
PATTERN_RADIUS_TOLERANCE = 0.4  # mm - tolerance for pattern similarity

# =============================================================================
# Hole Evidence Analysis Configuration
# =============================================================================

# Evidence weights for hole classification (must sum to 1.0)
HOLE_EVIDENCE_WEIGHTS = {
    "geometric_closure": 0.35,      # Complete circular geometry
    "concentric_nesting": 0.25,     # Inside other geometry
    "pattern_regularity": 0.20,     # Part of regular pattern
    "size_appropriateness": 0.15,   # Appropriate hole size
    "layer_context": 0.05          # Layer metadata hints
}

# Central position evidence
CENTRAL_EVIDENCE_HIGH_THRESHOLD = 0.8  # Threshold for central position bonus
CENTRAL_POSITION_BONUS = 0.15          # Bonus added for central holes
CENTRAL_CONFIDENCE_ADJUSTMENT = 0.7    # Lower threshold for confirmed central holes
CENTRAL_EVIDENCE_MIN_THRESHOLD = 0.95   # Minimum central evidence for adjustment

# Main body center detection bonus
MAIN_BODY_CENTER_TOLERANCE = 1.0       # Distance tolerance for main body center (mm)
MAIN_BODY_CENTER_BONUS = 0.3           # Large bonus for holes exactly at main body center

# Reconstruction confidence penalty
RECONSTRUCTION_CONFIDENCE_PENALTY_FACTOR = 0.2  # Penalty factor for reconstructed circles

# Geometric closure thresholds (degrees)
ARC_CLOSURE_EXCELLENT_THRESHOLD = 320.0  # Arcs > this get high closure evidence
ARC_CLOSURE_GOOD_THRESHOLD = 270.0       # Arcs > this get medium closure evidence
ARC_CLOSURE_EXCELLENT_EVIDENCE = 0.8     # Evidence score for excellent arcs
ARC_CLOSURE_GOOD_EVIDENCE = 0.5          # Evidence score for good arcs  
ARC_CLOSURE_POOR_EVIDENCE = 0.1          # Evidence score for poor arcs

# Concentric nesting evidence
NESTING_RELATIONSHIP_MIN_CONFIDENCE = 0.7  # Minimum relationship confidence

# Pattern regularity thresholds
PATTERN_LARGE_COUNT_THRESHOLD = 3        # 4+ total (including self) for max evidence  
PATTERN_MEDIUM_COUNT_THRESHOLD = 2       # 3 total for good evidence
PATTERN_LARGE_EVIDENCE = 1.0             # Evidence for large patterns
PATTERN_MEDIUM_EVIDENCE = 0.8            # Evidence for medium patterns  
PATTERN_SMALL_EVIDENCE = 0.6             # Evidence for small patterns
PATTERN_ISOLATED_EVIDENCE = 0.2          # Evidence for isolated features

# Spatial regularity analysis
SPATIAL_REGULARITY_MIN_CENTERS = 3       # Minimum centers needed for analysis
SPATIAL_REGULARITY_MIN_DISTANCES = 2     # Minimum distances for variance calc
SPATIAL_REGULARITY_MAX_VARIATION = 0.3   # Maximum relative deviation (30%)
SPATIAL_REGULARITY_BONUS = 0.1           # Bonus for regular spacing

# Size appropriateness evidence thresholds (mm)
HOLE_SIZE_VERY_SMALL_THRESHOLD = 2.0     # <= this for very small holes
HOLE_SIZE_SMALL_THRESHOLD = 6.0          # <= this for small holes  
HOLE_SIZE_MEDIUM_THRESHOLD = 12.0        # <= this for medium holes
HOLE_SIZE_LARGE_THRESHOLD = 18.0         # <= this for large holes

# Size appropriateness evidence scores
HOLE_SIZE_VERY_SMALL_EVIDENCE = 1.0      # Very small holes (screws, pins)
HOLE_SIZE_SMALL_EVIDENCE = 0.9           # Small-medium holes (bolts)
HOLE_SIZE_MEDIUM_EVIDENCE = 0.7          # Medium holes  
HOLE_SIZE_LARGE_EVIDENCE = 0.5           # Large holes
HOLE_SIZE_VERY_LARGE_EVIDENCE = 0.3      # Very large holes
HOLE_SIZE_OVERSIZED_EVIDENCE = 0.1       # Too large for typical holes

# Hole size categorization thresholds (mm)
HOLE_CATEGORY_VERY_SMALL_THRESHOLD = 1.5  # <= this for very_small
HOLE_CATEGORY_SMALL_THRESHOLD = 3.0       # <= this for small
HOLE_CATEGORY_MEDIUM_THRESHOLD = 5.0      # <= this for medium  
HOLE_CATEGORY_LARGE_THRESHOLD = 8.0       # <= this for large

# Layer context evidence scores
LAYER_HOLE_KEYWORDS_EVIDENCE = 0.9       # Explicit hole keywords
LAYER_FEATURE_KEYWORDS_EVIDENCE = 0.6    # Feature keywords
LAYER_CONSTRUCTION_KEYWORDS_EVIDENCE = 0.2  # Construction keywords  
LAYER_DEFAULT_EVIDENCE = 0.5             # Default/unknown layers

# Geometric closure evidence thresholds
CLOSURE_EVIDENCE_MIN_COVERAGE = 0.95  # For complete circles
CLOSURE_EVIDENCE_MIN_ARC_COVERAGE = 0.90  # For reconstructed circles

# Concentric nesting evidence thresholds  
CONCENTRIC_MIN_SIZE_RATIO = 0.1   # Inner/outer radius ratio
CONCENTRIC_MAX_SIZE_RATIO = 0.8   # Inner/outer radius ratio
CONCENTRIC_CENTER_TOLERANCE = 0.3  # mm

# =============================================================================
# Circle Detection Configuration
# =============================================================================

# Circle confidence calculation
CIRCLE_BASE_CONFIDENCE = 0.95                       # Base confidence for complete circle entities

# Size factor scoring for circles
CIRCLE_SIZE_FACTOR_IDEAL = 1.0                      # Factor for ideal-sized circles
CIRCLE_SIZE_FACTOR_ACCEPTABLE = 0.9                 # Factor for acceptable-sized circles
CIRCLE_SIZE_FACTOR_SMALL = 0.7                      # Factor for very small circles
CIRCLE_SIZE_FACTOR_LARGE = 0.6                      # Factor for very large circles
CIRCLE_SIZE_FACTOR_NO_RADIUS = 0.5                  # Factor when radius data unavailable

# Layer context factors
CIRCLE_LAYER_FACTOR_AUXILIARY = 0.8                 # Factor for construction/hidden layers
CIRCLE_LAYER_FACTOR_GEOMETRIC = 1.0                 # Factor for geometric feature layers
CIRCLE_LAYER_FACTOR_DEFAULT = 0.95                  # Factor for unlabeled layers

# Context analysis factors
CIRCLE_CONTEXT_FACTOR_NEUTRAL = 1.0                 # Default neutral confidence
CIRCLE_CONTEXT_FACTOR_COMPLEX = 0.95                # Slight penalty for complex geometry

# Size diversity analysis
CIRCLE_SIZE_DIVERSITY_THRESHOLD = 1.5               # Threshold for low vs high size diversity

# Main body detection thresholds  
CIRCLE_MAIN_BODY_CONSERVATIVE_THRESHOLD = 60.0      # Conservative threshold for reconstructed circles (mm)
CIRCLE_MAIN_BODY_CONCENTRIC_WEIGHT = 0.8            # Weight for concentric filtering
CIRCLE_ADAPTIVE_THRESHOLD_MULTIPLIER = 1.5          # Multiplier for statistical threshold
CIRCLE_RADIUS_FALLBACK_MULTIPLIER = 2.0             # Fallback multiplier when no variation
CIRCLE_MIN_SIZE_THRESHOLD = 15.0                    # Minimum absolute size threshold (mm)
CIRCLE_VERY_LARGE_THRESHOLD = 40.0                  # Threshold for very large circles (mm)
CIRCLE_CONCENTRIC_MIN_COUNT = 2                     # Minimum concentric circles for main body detection

# =============================================================================
# Through Hole Detection Enhanced Configuration  
# =============================================================================

# Central position analysis
THROUGH_HOLE_RADIUS_RATIO_THRESHOLD = 2.5           # Radius ratio for main body detection
THROUGH_HOLE_CENTRALITY_WEIGHT_POSITION = 0.6       # Weight for centrality score calculation
THROUGH_HOLE_CENTRALITY_WEIGHT_SIZE = 0.4           # Weight for size in centrality calculation
THROUGH_HOLE_CENTRAL_EVIDENCE_GOOD = 0.7            # Good central size evidence value
THROUGH_HOLE_CENTRAL_EVIDENCE_ACCEPTABLE = 0.6      # Acceptable central size evidence value
THROUGH_HOLE_CENTRAL_TOLERANCE_BASE = 5.0           # Base central tolerance (mm)
THROUGH_HOLE_CENTRAL_TOLERANCE_FACTOR = 0.1         # Factor of main body radius for tolerance

# Central hole size analysis
THROUGH_HOLE_SIZE_RATIO_MIN_GOOD = 0.5              # Minimum ratio for good central size
THROUGH_HOLE_SIZE_RATIO_MAX_GOOD = 1.2              # Maximum ratio for good central size  
THROUGH_HOLE_SIZE_RATIO_MIN_ACCEPTABLE = 0.3        # Minimum ratio for acceptable size
THROUGH_HOLE_SIZE_RATIO_MAX_ACCEPTABLE = 1.5        # Maximum ratio for acceptable size
THROUGH_HOLE_CENTRAL_SIZE_MIN = 1.0                 # Minimum central hole size (mm)
THROUGH_HOLE_CENTRAL_SIZE_MAX = 5.0                 # Maximum typical central hole size (mm)
THROUGH_HOLE_CENTRAL_EVIDENCE_MIN = 0.7             # Minimum evidence for central detection

# =============================================================================
# Square Hole Detection Configuration
# =============================================================================

# Side tolerance calculation
SQUARE_HOLE_SIDE_TOLERANCE_BASE = 0.5               # Base side tolerance (mm)
SQUARE_HOLE_SIDE_TOLERANCE_PERCENTAGE = 0.1         # Percentage tolerance for side matching

# Size constraints for square holes
SQUARE_HOLE_MIN_DIMENSION = 3.0                     # Minimum dimension for square holes (mm)
SQUARE_HOLE_MAX_MIN_DIMENSION = 50.0                # Maximum minimum dimension (mm)  
SQUARE_HOLE_MAX_MAX_DIMENSION = 100.0               # Maximum maximum dimension (mm)
SQUARE_HOLE_GOOD_SIZE_THRESHOLD = 5.0               # Threshold for good hole size (mm)

# Rectangle detection criteria
SQUARE_HOLE_ANGLE_TOLERANCE_MULTIPLIER = 2          # Multiplier for angle tolerance in rectangles
SQUARE_HOLE_MIN_RIGHT_ANGLES = 3                    # Minimum right angles required for rectangle
SQUARE_HOLE_MIN_MATCHING_PAIRS = 2                  # Minimum matching opposite side pairs required

# Confidence calculation weights
SQUARE_HOLE_BASE_CONFIDENCE_WEIGHT = 0.6            # Weight for base confidence
SQUARE_HOLE_EVIDENCE_WEIGHT = 0.4                   # Weight for evidence score

# Additional square hole parameters
SQUARE_HOLE_MIN_CONFIDENCE_THRESHOLD = 0.3          # Minimum confidence threshold for consideration
SQUARE_HOLE_BASE_CONFIDENCE_RECTANGULAR = 0.8       # Base confidence for rectangular shapes
SQUARE_HOLE_BASE_CONFIDENCE_NON_RECTANGULAR = 0.3   # Base confidence for non-rectangular shapes
SQUARE_HOLE_SQUARE_BONUS = 0.1                      # Confidence bonus for square shapes

# =============================================================================
# Additional Algorithmic Parameters
# =============================================================================

# Through hole pattern evidence thresholds
THROUGH_HOLE_PATTERN_EVIDENCE_THRESHOLD = 0.6       # Threshold for pattern evidence acceptance
THROUGH_HOLE_CIRCULAR_PATTERN_VARIATION = 0.3       # Maximum variation for circular pattern detection

# Arc angle constants (mathematical constants, not algorithmic)
ARC_ANGLE_HALF_CIRCLE = 180.0                       # Mathematical constant for half circle
ARC_ANGLE_FULL_CIRCLE = 360.0                       # Mathematical constant for full circle
ARC_GAP_OVERLAP_TOLERANCE = 0.1                     # Tolerance for arc gap/overlap detection

# Geometric overlap estimation
GEOMETRIC_OVERLAP_CONSERVATIVE_ESTIMATE = 0.1       # Conservative estimate factor for overlap size
GEOMETRIC_OVERLAP_MERGE_TOLERANCE_FACTOR = 0.5      # Factor for merging tolerance in overlap detection

# High confidence relationship threshold
HIGH_CONFIDENCE_RELATIONSHIP_THRESHOLD = 0.8        # Threshold for high confidence relationships

# Feature type analysis
FEATURE_HIGH_CONFIDENCE_THRESHOLD = 0.8             # Threshold for high confidence features
THROUGH_HOLE_PREFERENCE_CONFIDENCE = 0.7            # Confidence threshold for preferring holes over circles

# Expected feature builder confidence binning
CONFIDENCE_BIN_VERY_LOW_MAX = 0.2                   # Maximum for very low confidence bin
CONFIDENCE_BIN_LOW_MAX = 0.4                        # Maximum for low confidence bin  
CONFIDENCE_BIN_MEDIUM_MAX = 0.6                     # Maximum for medium confidence bin
CONFIDENCE_BIN_HIGH_MAX = 0.8                       # Maximum for high confidence bin

# Percentile calculation factors (for statistical analysis)
PERCENTILE_75_FACTOR = 0.75                         # 75th percentile factor
PERCENTILE_90_FACTOR = 0.9                          # 90th percentile factor

# Geometry reconstruction legacy threshold (for geometry_reconstruction_old.py)
LEGACY_RECONSTRUCTION_CONFIDENCE_THRESHOLD = 0.5     # Legacy confidence threshold

# =============================================================================
# Additional Algorithmic Parameters Found in Forensic Audit
# =============================================================================

# Arc compatibility confidence calculation
ARC_COMPATIBILITY_BASE_CONFIDENCE = 0.9             # Base confidence for arc compatibility
ARC_COMPATIBILITY_GAP_PENALTY_FACTOR = 0.3          # Penalty factor for angular gaps

# Through hole detection contextual thresholds
THROUGH_HOLE_CIRCULAR_FILE_MIN_ENTITIES = 5         # Minimum circular entities to trigger special central detection
THROUGH_HOLE_SPATIAL_MIN_CENTERS = 3               # Minimum centers for spatial analysis
THROUGH_HOLE_SPATIAL_MIN_DISTANCES = 3             # Minimum distances for pattern analysis
THROUGH_HOLE_CIRCULAR_PATTERN_MIN_CENTERS = 4      # Minimum centers for circular pattern analysis

# Expected feature builder heuristics
EXPECTED_FEATURE_CANDIDATE_MULTIPLIER = 2          # Multiplier for candidate count estimation

# Entity models geometric validation
FULL_CIRCLE_ARC_SPAN_TOLERANCE = 1.0              # Tolerance for detecting full circle arcs (degrees)

# Parser geometric precision
GEOMETRIC_COORDINATE_PRECISION = 6                 # Decimal places for coordinate normalization
GEOMETRIC_SIZE_PRECISION = 6                      # Decimal places for size normalization
GEOMETRIC_ANGLE_PRECISION = 6                     # Decimal places for angle normalization

# Weight validation tolerance
WEIGHT_VALIDATION_TOLERANCE = 0.001               # Tolerance for weight sum validation

# Arc group default tolerance (geometry reconstruction)
ARC_GROUP_DEFAULT_TOLERANCE = 1.0                 # Default tolerance when not specified

# Additional through hole detection thresholds
THROUGH_HOLE_SIZE_SCORE_NORMALIZATION = 100.0       # Normalization factor for size scores
THROUGH_HOLE_DEFAULT_EVIDENCE = 0.5                 # Default evidence when data insufficient
THROUGH_HOLE_NON_CENTRAL_EVIDENCE = 0.3             # Evidence for non-central position
THROUGH_HOLE_POOR_SIZE_EVIDENCE = 0.4               # Evidence for poor central size
THROUGH_HOLE_SPATIAL_TOLERANCE_FACTOR_HIGH = 0.3    # 30% of average distance factor
THROUGH_HOLE_SPATIAL_TOLERANCE_FACTOR_LOW = 0.15    # 15% of maximum distance factor
THROUGH_HOLE_CIRCULAR_PATTERN_BASE_EVIDENCE = 0.9   # Base evidence for circular patterns

# Square hole evidence scoring
SQUARE_HOLE_EVIDENCE_GEOMETRIC_CLOSURE = 0.3        # Weight for geometric closure
SQUARE_HOLE_EVIDENCE_SIZE_APPROPRIATENESS = 0.4     # Weight for size appropriateness
SQUARE_HOLE_EVIDENCE_CONTEXT = 0.2                  # Weight for context evidence
SQUARE_HOLE_EVIDENCE_LAYER = 0.1                    # Weight for layer evidence
SQUARE_HOLE_EVIDENCE_CONTEXT_MODERATE = 0.7         # Moderate context evidence value
SQUARE_HOLE_EVIDENCE_SIZE_GOOD = 1.0                # Good size evidence
SQUARE_HOLE_EVIDENCE_SIZE_SMALLER = 0.8             # Smaller holes evidence
SQUARE_HOLE_EVIDENCE_SIZE_POOR = 0.3                # Poor size evidence
SQUARE_HOLE_EVIDENCE_LAYER_POOR = 0.2               # Poor layer evidence
SQUARE_HOLE_EVIDENCE_LAYER_NEUTRAL = 0.5            # Neutral layer evidence
SQUARE_HOLE_EVIDENCE_LAYER_GOOD = 0.9               # Good layer evidence (hole keywords)

# Semantic grouping confidence boost parameters
SEMANTIC_GROUPING_CONFIDENCE_BOOST_MAX = 0.1        # Maximum confidence boost
SEMANTIC_GROUPING_CONFIDENCE_BOOST_PER_FEATURE = 0.02  # Boost per additional feature

# Hardcoded comparison thresholds
SEMANTIC_GROUP_MIN_RADII_FOR_SPAN = 1               # Minimum radii needed for span calculation
SEMANTIC_GROUP_MIN_FEATURES_FOR_BOOST = 1           # Minimum features needed for confidence boost

# Position scoring defaults
SEMANTIC_DEFAULT_POSITION_SCORE = 0.5               # Default position score when insufficient data

# Significance filter thresholds
SIGNIFICANCE_FILTER_MIN_FEATURES_FOR_DEDUP = 1      # Minimum features needed for deduplication
SIGNIFICANCE_FILTER_PERCENTAGE_MULTIPLIER = 100     # Multiplier for percentage calculation (100%)

# Square hole detection specific constants
SQUARE_HOLE_MIN_LINE_ENTITIES = 4                   # Minimum LINE entities needed for square detection
SQUARE_HOLE_MIN_SIDES_RECTANGLE = 4                 # Minimum sides required for rectangle
SQUARE_HOLE_MIN_SIDES_POLYGON = 3                   # Minimum sides for general polygon
SQUARE_HOLE_MAX_CHAIN_LENGTH = 20                   # Maximum chain length (prevents infinite loops)
SQUARE_HOLE_MIN_VERTICES = 4                        # Minimum vertices for rectangle analysis

# Expected feature builder proximity threshold
EXPECTED_FEATURE_PROXIMITY_THRESHOLD = 20.0         # Proximity threshold for relationships (mm)

# =============================================================================
# Semantic Feature Grouping Configuration
# =============================================================================

# Geometric relationship analysis thresholds
SEMANTIC_NESTING_RATIO_THRESHOLD = 1.5          # Minimum radius ratio for nested classification
SEMANTIC_ADJACENCY_TOLERANCE = 2.0              # mm - tolerance for geometric adjacency detection
SEMANTIC_CONTAINMENT_TOLERANCE = 1.0            # mm - tolerance for containment analysis

# Feature selection preferences for grouped features
SEMANTIC_HOLE_SIZE_IDEAL_MIN = 2.0               # mm - minimum preferred hole size
SEMANTIC_HOLE_SIZE_IDEAL_MAX = 15.0              # mm - maximum preferred hole size  
SEMANTIC_HOLE_SIZE_ACCEPTABLE_MIN = 1.0          # mm - minimum acceptable hole size
SEMANTIC_HOLE_SIZE_ACCEPTABLE_MAX = 25.0         # mm - maximum acceptable hole size
SEMANTIC_HOLE_SIZE_BODY_THRESHOLD = 25.0         # mm - holes larger than this may be body geometry

SEMANTIC_CIRCLE_SIZE_SMALL_MAX = 10.0            # mm - maximum for small circles (preferred)
SEMANTIC_CIRCLE_SIZE_MEDIUM_MAX = 20.0           # mm - maximum for medium circles
SEMANTIC_CIRCLE_SIZE_LARGE_MAX = 35.0            # mm - maximum for large circles
SEMANTIC_CIRCLE_SIZE_BODY_THRESHOLD = 35.0       # mm - circles larger than this are likely body geometry

# Circle size classification lower bound
SEMANTIC_CIRCLE_SIZE_MIN_INSPECTION = 1.0        # mm - minimum radius for inspection circle classification

# Hole size scoring values (algorithmic scoring weights)
SEMANTIC_HOLE_SIZE_SCORE_IDEAL = 3               # Score for ideal inspection hole sizes
SEMANTIC_HOLE_SIZE_SCORE_ACCEPTABLE = 2          # Score for acceptable hole sizes
SEMANTIC_HOLE_SIZE_SCORE_SMALL = 1               # Score for small holes (construction details)
SEMANTIC_HOLE_SIZE_SCORE_LARGE = 0.5             # Score for large holes (possibly body geometry)

# Circle size scoring values (algorithmic scoring weights)
SEMANTIC_CIRCLE_SIZE_SCORE_SMALL = 3             # Score for small-medium circles (likely inspection)
SEMANTIC_CIRCLE_SIZE_SCORE_MEDIUM = 2.5          # Score for medium-large circles
SEMANTIC_CIRCLE_SIZE_SCORE_LARGE = 1             # Score for large circles (possibly body boundaries)
SEMANTIC_CIRCLE_SIZE_SCORE_VERY_LARGE = 0.5      # Score for very large or very small circles

# Circle source type scoring values (algorithmic source reliability weights)
SEMANTIC_CIRCLE_SOURCE_SCORE_EXPLICIT = 1.2      # Score for explicit circle entities
SEMANTIC_CIRCLE_SOURCE_SCORE_RECONSTRUCTED = 0.8 # Score for reconstructed circles
SEMANTIC_CIRCLE_SOURCE_SCORE_OTHER = 1.0         # Score for other source types

# Concentric series analysis thresholds
SEMANTIC_CONCENTRIC_MIN_COUNT_FOR_POSITION = 2   # Minimum count in series to analyze position scoring
SEMANTIC_GEOMETRIC_CONTEXT_MIN_RADII = 2         # Minimum radii needed for geometric context analysis
SEMANTIC_PERCENTILE_MIN_COUNT_75 = 2             # Minimum count for 75th percentile calculation  
SEMANTIC_PERCENTILE_MIN_COUNT_90 = 9             # Minimum count for 90th percentile calculation

# Confidence scoring weights for grouped feature selection
SEMANTIC_SIZE_WEIGHT = 0.6                       # Weight for size appropriateness in selection
SEMANTIC_EVIDENCE_WEIGHT = 0.3                  # Weight for evidence quality in selection  
SEMANTIC_POSITION_WEIGHT = 0.1                  # Weight for position in concentric series

SEMANTIC_CIRCLE_SIZE_WEIGHT = 0.7                # Weight for size appropriateness in circle selection
SEMANTIC_CIRCLE_CONFIDENCE_WEIGHT = 0.2         # Weight for confidence in circle selection
SEMANTIC_CIRCLE_SOURCE_WEIGHT = 0.1             # Weight for source type in circle selection

# Body geometry classification thresholds
SEMANTIC_BODY_ARC_COUNT_THRESHOLD = 6            # Minimum arc count for body geometry classification

# Body geometry classification confidence scores
SEMANTIC_BODY_RELATIVE_SIZE_CONFIDENCE = 0.8     # Confidence for relative size classification
SEMANTIC_BODY_ABSOLUTE_SIZE_CONFIDENCE = 0.9     # Confidence for absolute size classification  
SEMANTIC_BODY_RECONSTRUCTION_CONFIDENCE = 0.7    # Confidence for reconstruction-based classification
SEMANTIC_INSPECTION_FEATURE_CONFIDENCE = 0.6     # Confidence for inspection feature classification

# =============================================================================
# Geometric Relationship Analysis Configuration
# =============================================================================

# Configuration for analyzing spatial relationships between entities
RELATIONSHIP_CONFIG = {
    # Distance within which entities are considered "nearby"
    "proximity_threshold": 8.0,  # mm
    
    # Threshold for detecting repeated patterns
    "pattern_detection_tolerance": 1.0,  # mm
    
    # Minimum number of similar features to constitute a pattern
    "min_pattern_count": PATTERN_MIN_SIMILAR_COUNT,
}

# =============================================================================
# Visualization Configuration  
# =============================================================================

# Plot margin calculation
VISUALIZATION_MARGIN_BASE_LARGE = 10.0              # Base margin for large plots (mm)
VISUALIZATION_MARGIN_BASE_SMALL = 5.0               # Base margin for small plots (mm)
VISUALIZATION_MARGIN_PERCENTAGE = 0.05              # Percentage-based margin (5% of plot bounds)

# Alpha (transparency) values for plot elements
VISUALIZATION_ALPHA_DXF_GEOMETRY = 0.7               # Alpha for DXF reference geometry
VISUALIZATION_ALPHA_RECONSTRUCTED_GEOMETRY = 0.6    # Alpha for reconstructed geometry  
VISUALIZATION_ALPHA_FEATURE_BASE = 0.6               # Base alpha for features
VISUALIZATION_ALPHA_FEATURE_CONFIDENCE_FACTOR = 0.4 # Factor for confidence-based alpha adjustment

# Line widths for different elements
VISUALIZATION_LINEWIDTH_DXF = 1.0                   # Line width for DXF geometry
VISUALIZATION_LINEWIDTH_RECONSTRUCTED = 1.5         # Line width for reconstructed geometry

# Feature label positioning
VISUALIZATION_LABEL_OFFSET_BASE = 5.0               # Base offset for feature labels (mm)
VISUALIZATION_LABEL_OFFSET_MIN = 8.0                # Minimum offset for feature labels (mm)
VISUALIZATION_LABEL_OFFSET_FACTOR_X = 0.7           # X-direction offset factor
VISUALIZATION_LABEL_OFFSET_FACTOR_Y = 0.7           # Y-direction offset factor

# Default feature dimensions (for visualization when data missing)
VISUALIZATION_DEFAULT_FEATURE_WIDTH = 10.0          # Default width for square features (mm)
VISUALIZATION_DEFAULT_FEATURE_HEIGHT = 10.0         # Default height for square features (mm)

# Arc angle handling
VISUALIZATION_ARC_ANGLE_FULL_CIRCLE = 360.0         # Full circle angle for arc wraparound

# Multipliers for various visualization calculations
VISUALIZATION_THETA_SAMPLES_MULTIPLIER = 2          # Multiplier for theta sample calculation
VISUALIZATION_RADIUS_ARC_MULTIPLIER = 2             # Multiplier for arc radius calculations

# Additional visualization constants
VISUALIZATION_GRID_ALPHA = 0.2                      # Alpha for grid display
VISUALIZATION_DEFAULT_FONT_SIZE = 10                 # Default font size for titles
VISUALIZATION_LABEL_FONT_SIZE = 8                   # Font size for feature labels
VISUALIZATION_DEFAULT_BOUNDS = 10                    # Default coordinate bounds when no geometry
VISUALIZATION_BBOX_ALPHA = 0.9                      # Alpha for label bounding boxes
VISUALIZATION_ARROW_ALPHA = 0.8                     # Alpha for arrows
VISUALIZATION_LEGEND_ALPHA = 0.9                    # Alpha for legend background
VISUALIZATION_BBOX_PAD_NORMAL = 0.3                 # Padding for normal bounding boxes
VISUALIZATION_BBOX_PAD_LARGE = 0.5                  # Padding for large bounding boxes

# Text positioning constants (UI layout, not algorithmic)
VISUALIZATION_TEXT_CENTER_X = 0.5                   # Center X coordinate for text positioning
VISUALIZATION_TEXT_CENTER_Y = 0.5                   # Center Y coordinate for text positioning  
VISUALIZATION_TEXT_CORNER_OFFSET = 0.02             # Corner offset for text positioning

VISUALIZATION_CONFIG = {
    # Figure size for feature visualization (inches)
    "figure_size": (12, 8),
    
    # DPI for saved visualizations
    "dpi": 150,
    
    # Colors for different feature types
    "colors": {
        "circle": "blue",
        "through_hole": "red", 
        "arc": "green",
        "line": "gray",
        "reconstructed": "purple"
    },
    
    # Marker sizes
    "marker_sizes": {
        "feature_center": 8,
        "entity_point": 4
    }
}

# =============================================================================
# Debug/Diagnostic Configuration
# =============================================================================

DEBUG_CONFIG = {
    # Enable detailed logging for geometry reconstruction
    "log_reconstruction": True,
    
    # Enable detailed logging for feature detection
    "log_feature_detection": True,
    
    # Enable detailed logging for significance filtering
    "log_significance_filter": True,
    
    # Save intermediate processing results
    "save_intermediate_results": False
}

def get_config() -> Dict[str, Any]:
    """Get the complete configuration dictionary."""
    return {
        "geometric_tolerance": GEOMETRIC_TOLERANCE,
        "angular_tolerance": ANGULAR_TOLERANCE, 
        "radius_tolerance_absolute": RADIUS_TOLERANCE_ABSOLUTE,
        "radius_tolerance_relative": RADIUS_TOLERANCE_RELATIVE,
        "concentricity_tolerance": CONCENTRICITY_TOLERANCE,
        "circle_reconstruction_min_coverage": CIRCLE_RECONSTRUCTION_MIN_COVERAGE,
        "circle_reconstruction_max_gap": CIRCLE_RECONSTRUCTION_MAX_GAP,
        "circle_min_radius": CIRCLE_MIN_RADIUS,
        "circle_max_radius": CIRCLE_MAX_RADIUS,
        "through_hole_min_radius": THROUGH_HOLE_MIN_RADIUS,
        "through_hole_max_radius": THROUGH_HOLE_MAX_RADIUS,
        "min_feature_confidence": MIN_FEATURE_CONFIDENCE,
        "significance_min_radius": SIGNIFICANCE_MIN_RADIUS,
        "relationship_config": RELATIONSHIP_CONFIG,
        "visualization_config": VISUALIZATION_CONFIG,
        "debug_config": DEBUG_CONFIG
    }