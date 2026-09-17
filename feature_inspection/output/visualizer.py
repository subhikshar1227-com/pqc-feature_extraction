"""
Actual Feature Visualization

Creates visual overlays showing detected actual features on original product images.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List
from pathlib import Path

from feature_extraction.expected.feature_types import FeatureType, Point2D
from ..models.actual_feature import ActualFeature, ActualFeatureSet
from ..config import (
    COLOR_ACTUAL_FEATURE, COLOR_MATCHED_FEATURE, COLOR_EXTRA_FEATURE,
    VISUALIZATION_CIRCLE_THICKNESS, VISUALIZATION_TEXT_FONT_SCALE,
    VISUALIZATION_TEXT_THICKNESS, VISUALIZATION_MARKER_SIZE
)

logger = logging.getLogger(__name__)


class ActualFeatureVisualizer:
    """
    Visualizes detected actual features on original product images.
    
    Creates annotated images showing feature locations, types, confidence,
    and other metadata for debugging and inspection purposes.
    """
    
    def __init__(self):
        """Initialize actual feature visualizer."""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def visualize_actual_features(self, 
                                original_image_path: Path,
                                actual_feature_set: ActualFeatureSet,
                                output_path: Optional[Path] = None,
                                matched_feature_ids: Optional[List[str]] = None) -> Path:
        """
        Create visualization of actual features on original image.
        
        Args:
            original_image_path: Path to original product image
            actual_feature_set: Set of detected actual features
            output_path: Where to save visualization. If None, saves next to original image
            matched_feature_ids: List of feature IDs that were successfully matched
            
        Returns:
            Path to saved visualization
        """
        logger.info(f"Creating actual feature visualization for {len(actual_feature_set.features)} features")
        
        # Load original image
        if not original_image_path.exists():
            raise FileNotFoundError(f"Original image not found: {original_image_path}")
        
        original_image = cv2.imread(str(original_image_path))
        if original_image is None:
            raise ValueError(f"Failed to load image: {original_image_path}")
        
        # Create visualization image
        viz_image = original_image.copy()
        
        # Determine output path if not provided
        if output_path is None:
            output_path = original_image_path.parent / f"{original_image_path.stem}_actual_features.png"
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Handle coordinate system - visualization must be in image pixels
        features_to_visualize = self._prepare_features_for_visualization(
            actual_feature_set, original_image.shape[:2]
        )
        
        matched_ids = set(matched_feature_ids) if matched_feature_ids else set()
        
        # Draw features
        for feature in features_to_visualize:
            self._draw_feature(viz_image, feature, matched_ids)
        
        # Add legend and metadata
        self._add_visualization_metadata(viz_image, actual_feature_set, len(matched_ids))
        
        # Save visualization
        cv2.imwrite(str(output_path), viz_image)
        
        logger.info(f"Actual feature visualization saved to {output_path}")
        return output_path
    
    def _prepare_features_for_visualization(self, 
                                          feature_set: ActualFeatureSet, 
                                          image_dimensions: Tuple[int, int]) -> List[ActualFeature]:
        """
        Prepare features for visualization by ensuring they're in image pixel coordinates.
        
        Args:
            feature_set: Feature set to visualize
            image_dimensions: (height, width) of the target image
            
        Returns:
            List of features in image pixel coordinates
        """
        if feature_set.coordinate_system == "image_pixels":
            # Features are already in image pixels
            return feature_set.features
        
        elif feature_set.coordinate_system.startswith("image_pixels"):
            # Features are in image pixels (with some error condition)
            logger.warning(f"Feature coordinate system: {feature_set.coordinate_system}")
            return feature_set.features
        
        elif feature_set.coordinate_system == "dxf_mm":
            # Features are in DXF coordinates - cannot visualize directly
            logger.error("Cannot visualize features in DXF coordinates without inverse transformation")
            logger.error("Visualization requires features in image pixel coordinates")
            
            # Return empty list rather than incorrect visualization
            return []
        
        else:
            logger.warning(f"Unknown coordinate system: {feature_set.coordinate_system}")
            return feature_set.features
    
    def _draw_feature(self, 
                     image: np.ndarray, 
                     feature: ActualFeature,
                     matched_ids: set) -> None:
        """
        Draw a single feature on the image.
        
        Args:
            image: Image to draw on (modified in place)
            feature: Feature to draw
            matched_ids: Set of matched feature IDs
        """
        # Determine color based on match status
        if feature.feature_id in matched_ids:
            color = COLOR_MATCHED_FEATURE  # Yellow for matched
        else:
            color = COLOR_EXTRA_FEATURE   # Magenta for unmatched
        
        # Get feature center in integer coordinates
        center_x = int(round(feature.center.x))
        center_y = int(round(feature.center.y))
        center_point = (center_x, center_y)
        
        # Draw feature geometry
        if feature.feature_type in [FeatureType.CIRCLE, FeatureType.THROUGH_HOLE]:
            # Draw circle
            if feature.radius is not None:
                radius = int(round(feature.radius))
                cv2.circle(image, center_point, radius, color, VISUALIZATION_CIRCLE_THICKNESS)
            
            # Draw center marker
            cv2.circle(image, center_point, VISUALIZATION_MARKER_SIZE, color, -1)
        
        elif feature.feature_type in [FeatureType.RECTANGULAR_HOLE, FeatureType.SQUARE_HOLE]:
            # Draw rectangle
            if feature.width is not None and feature.height is not None:
                half_width = int(round(feature.width / 2))
                half_height = int(round(feature.height / 2))
                
                top_left = (center_x - half_width, center_y - half_height)
                bottom_right = (center_x + half_width, center_y + half_height)
                
                cv2.rectangle(image, top_left, bottom_right, color, VISUALIZATION_CIRCLE_THICKNESS)
            
            # Draw center marker
            cv2.circle(image, center_point, VISUALIZATION_MARKER_SIZE, color, -1)
        
        # Draw feature label
        self._draw_feature_label(image, feature, center_point, color)
    
    def _draw_feature_label(self, 
                          image: np.ndarray,
                          feature: ActualFeature, 
                          center_point: Tuple[int, int],
                          color: Tuple[int, int, int]) -> None:
        """
        Draw feature ID and metadata label.
        
        Args:
            image: Image to draw on
            feature: Feature to label
            center_point: Center coordinates
            color: Label color
        """
        # Create label text
        feature_type_short = {
            FeatureType.CIRCLE: "C",
            FeatureType.THROUGH_HOLE: "H",
            FeatureType.RECTANGULAR_HOLE: "R", 
            FeatureType.SQUARE_HOLE: "S"
        }.get(feature.feature_type, "?")
        
        confidence_str = f"{feature.confidence:.2f}"
        label_text = f"{feature_type_short}:{confidence_str}"
        
        # Position label above the feature
        label_x = center_point[0] - 20
        label_y = center_point[1] - 15
        
        # Ensure label stays within image bounds
        label_x = max(5, min(label_x, image.shape[1] - 80))
        label_y = max(20, label_y)
        
        # Draw label background
        text_size = cv2.getTextSize(label_text, self.font, VISUALIZATION_TEXT_FONT_SCALE, VISUALIZATION_TEXT_THICKNESS)[0]
        background_top_left = (label_x - 2, label_y - text_size[1] - 2)
        background_bottom_right = (label_x + text_size[0] + 2, label_y + 2)
        cv2.rectangle(image, background_top_left, background_bottom_right, (0, 0, 0), -1)
        
        # Draw label text
        cv2.putText(image, label_text, (label_x, label_y), self.font, 
                   VISUALIZATION_TEXT_FONT_SCALE, color, VISUALIZATION_TEXT_THICKNESS)
    
    def _add_visualization_metadata(self, 
                                  image: np.ndarray,
                                  feature_set: ActualFeatureSet,
                                  matched_count: int) -> None:
        """
        Add metadata overlay to the visualization.
        
        Args:
            image: Image to add metadata to
            feature_set: Feature set being visualized
            matched_count: Number of matched features
        """
        # Prepare metadata text
        metadata_lines = [
            f"Detected: {len(feature_set.features)} features",
            f"Matched: {matched_count}",
            f"Coord Sys: {feature_set.coordinate_system}",
            f"Avg Conf: {feature_set.average_confidence:.3f}",
            f"Timestamp: {feature_set.detection_timestamp}"
        ]
        
        # Position metadata in top-left corner
        start_y = 30
        line_height = 25
        
        for i, line in enumerate(metadata_lines):
            y_position = start_y + i * line_height
            
            # Draw background rectangle
            text_size = cv2.getTextSize(line, self.font, 0.6, 1)[0]
            cv2.rectangle(image, (5, y_position - text_size[1] - 5), 
                         (text_size[0] + 10, y_position + 5), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(image, line, (10, y_position), self.font, 0.6, (255, 255, 255), 1)
        
        # Add legend in bottom-right corner
        self._add_color_legend(image)
    
    def _add_color_legend(self, image: np.ndarray) -> None:
        """
        Add color legend to the visualization.
        
        Args:
            image: Image to add legend to
        """
        legend_items = [
            ("Matched", COLOR_MATCHED_FEATURE),
            ("Unmatched", COLOR_EXTRA_FEATURE)
        ]
        
        # Position legend in bottom-right corner
        legend_x = image.shape[1] - 150
        legend_y = image.shape[0] - 60
        
        # Draw legend background
        cv2.rectangle(image, (legend_x - 10, legend_y - 40), 
                     (legend_x + 140, legend_y + 20), (0, 0, 0), -1)
        
        # Draw legend items
        for i, (label, color) in enumerate(legend_items):
            y_pos = legend_y - 25 + i * 20
            
            # Draw color sample
            cv2.circle(image, (legend_x, y_pos), 8, color, -1)
            
            # Draw label
            cv2.putText(image, label, (legend_x + 15, y_pos + 5), self.font, 0.5, (255, 255, 255), 1)
    
    def create_comparison_visualization(self,
                                     original_image_path: Path,
                                     actual_features: ActualFeatureSet,
                                     matched_feature_ids: List[str],
                                     output_path: Path) -> Path:
        """
        Create visualization showing matched vs unmatched actual features.
        
        Args:
            original_image_path: Path to original image
            actual_features: Detected actual features
            matched_feature_ids: IDs of features that were matched
            output_path: Where to save the visualization
            
        Returns:
            Path to saved visualization
        """
        return self.visualize_actual_features(
            original_image_path, actual_features, output_path, matched_feature_ids
        )