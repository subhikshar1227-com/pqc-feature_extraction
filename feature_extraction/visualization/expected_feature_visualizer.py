"""
Expected Feature Visualizer

Creates visualizations showing complete DXF geometry with highlighted expected features.
The visualization displays the complete CAD reference geometry with feature overlays.
"""

import logging
import math
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Arc, Rectangle
import numpy as np

from ..expected.feature_types import ExpectedFeatureSet, ExpectedFeature, FeatureType
from ..dxf.entity_models import NormalizedEntity, EntityType
from ..config import (
    VISUALIZATION_MARGIN_BASE_LARGE, VISUALIZATION_MARGIN_BASE_SMALL, VISUALIZATION_MARGIN_PERCENTAGE,
    VISUALIZATION_ALPHA_DXF_GEOMETRY, VISUALIZATION_ALPHA_RECONSTRUCTED_GEOMETRY,
    VISUALIZATION_ALPHA_FEATURE_BASE, VISUALIZATION_ALPHA_FEATURE_CONFIDENCE_FACTOR,
    VISUALIZATION_LINEWIDTH_DXF, VISUALIZATION_LINEWIDTH_RECONSTRUCTED,
    VISUALIZATION_LABEL_OFFSET_BASE, VISUALIZATION_LABEL_OFFSET_MIN,
    VISUALIZATION_LABEL_OFFSET_FACTOR_X, VISUALIZATION_LABEL_OFFSET_FACTOR_Y,
    VISUALIZATION_DEFAULT_FEATURE_WIDTH, VISUALIZATION_DEFAULT_FEATURE_HEIGHT,
    VISUALIZATION_ARC_ANGLE_FULL_CIRCLE, VISUALIZATION_THETA_SAMPLES_MULTIPLIER,
    VISUALIZATION_RADIUS_ARC_MULTIPLIER, VISUALIZATION_CONFIG,
    # Additional visualization constants
    VISUALIZATION_GRID_ALPHA, VISUALIZATION_DEFAULT_FONT_SIZE, VISUALIZATION_LABEL_FONT_SIZE, VISUALIZATION_DEFAULT_BOUNDS,
    VISUALIZATION_BBOX_ALPHA, VISUALIZATION_ARROW_ALPHA, VISUALIZATION_LEGEND_ALPHA,
    VISUALIZATION_BBOX_PAD_NORMAL, VISUALIZATION_BBOX_PAD_LARGE,
    # Text positioning constants
    VISUALIZATION_TEXT_CENTER_X, VISUALIZATION_TEXT_CENTER_Y, VISUALIZATION_TEXT_CORNER_OFFSET
)

logger = logging.getLogger(__name__)


class ExpectedFeatureVisualizer:
    """Visualizes complete DXF geometry with highlighted expected features."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize visualizer.
        
        Args:
            config: Optional visualization configuration override
        """
        self.config = config or VISUALIZATION_CONFIG.copy()
        
        # Enhanced color scheme for complete DXF visualization
        self.dxf_colors = {
            "line": "#CCCCCC",           # Light gray for reference lines
            "arc": "#CCCCCC",            # Light gray for reference arcs  
            "circle": "#CCCCCC",         # Light gray for reference circles
            "reconstructed": "#DDDDDD"   # Slightly darker gray for reconstructed geometry
        }
        
        self.feature_colors = {
            "circle": "#2E86C1",         # Blue for circle features
            "through_hole": "#E74C3C",   # Red for hole features
            "square_hole": "#E74C3C"     # Red for square holes
        }
        
    def visualize_feature_set(self, feature_set: ExpectedFeatureSet, 
                            output_path: Optional[Path] = None,
                            show_entities: bool = True,
                            show_labels: bool = True) -> Path:
        """
        Create a comprehensive visualization showing complete DXF geometry with highlighted features.
        
        Args:
            feature_set: The feature set to visualize
            output_path: Optional output file path (defaults to auto-generated)
            show_entities: Whether to show underlying DXF entities
            show_labels: Whether to show feature labels
            
        Returns:
            Path to the saved visualization file
        """
        logger.info(f"Creating complete DXF visualization for {feature_set.total_feature_count} features")
        
        # Create figure
        fig_size = self.config["figure_size"]
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        
        # Calculate plot bounds from COMPLETE geometry (not just features)
        bounds = self._calculate_complete_bounds(feature_set)
        
        # Set up the plot with appropriate margins
        margin = max(VISUALIZATION_MARGIN_BASE_LARGE, (bounds["max_x"] - bounds["min_x"]) * VISUALIZATION_MARGIN_PERCENTAGE)
        ax.set_xlim(bounds["min_x"] - margin, bounds["max_x"] + margin)
        ax.set_ylim(bounds["min_y"] - margin, bounds["max_y"] + margin)
        ax.set_aspect('equal')
        ax.grid(True, alpha=VISUALIZATION_GRID_ALPHA)
        
        # Step 1: Plot complete DXF reference geometry (background)
        if show_entities and feature_set.reference_geometry:
            self._plot_reference_geometry(ax, feature_set.reference_geometry)
        
        # Step 2: Plot reconstructed geometry (intermediate)
        if feature_set.reconstructed_geometry and hasattr(feature_set.reconstructed_geometry, 'reconstructed_circles'):
            self._plot_reconstructed_geometry(ax, feature_set.reconstructed_geometry)
        
        # Step 3: Plot expected features (foreground highlights)
        self._plot_expected_features(ax, feature_set.features, show_labels)
        
        # Set title
        title = f"Complete DXF Geometry + Expected Features: {feature_set.source_dxf_path.name}"
        title += f"\n{feature_set.total_feature_count} features "
        title += f"({feature_set.circle_count} circles, {feature_set.through_hole_count} holes)"
        ax.set_title(title)
        
        # Add comprehensive legend
        self._add_comprehensive_legend(ax)
        
        # Add metadata text
        self._add_metadata_text(ax, feature_set)
        
        # Save the visualization
        if output_path is None:
            output_dir = Path("outputs") / feature_set.source_dxf_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{feature_set.source_dxf_path.stem}_expected_features.png"
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config["dpi"], bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved complete DXF + features visualization: {output_path}")
        return output_path
    
    def create_feature_comparison(self, feature_sets: Dict[str, ExpectedFeatureSet],
                                output_path: Optional[Path] = None) -> Path:
        """Create a comparison visualization with complete DXF geometry for multiple feature sets."""
        
        logger.info(f"Creating complete DXF comparison for {len(feature_sets)} feature sets")
        
        # Determine subplot layout
        n_sets = len(feature_sets)
        cols = min(2, n_sets)
        rows = math.ceil(n_sets / cols)
        
        fig_width = self.config["figure_size"][0] * cols
        fig_height = self.config["figure_size"][1] * rows
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        
        if n_sets == 1:
            axes = [axes]
        elif rows == 1:
            axes = list(axes)
        else:
            axes = axes.flatten()
        
        # Plot each feature set with complete geometry
        for i, (name, feature_set) in enumerate(feature_sets.items()):
            if i < len(axes):
                ax = axes[i]
                self._plot_single_complete_feature_set(ax, feature_set, title=name)
        
        # Hide unused subplots
        for i in range(n_sets, len(axes)):
            axes[i].set_visible(False)
        
        # Save comparison
        if output_path is None:
            output_path = Path("outputs") / "expected_features_comparison.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config["dpi"], bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved complete DXF comparison: {output_path}")
        return output_path
        
    def _plot_single_complete_feature_set(self, ax, feature_set: ExpectedFeatureSet, title: str = None):
        """Plot a single complete feature set with DXF geometry on the given axis."""
        
        if not feature_set.features and not feature_set.reference_geometry:
            ax.text(VISUALIZATION_TEXT_CENTER_X, VISUALIZATION_TEXT_CENTER_Y, "No geometry detected", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title or feature_set.source_dxf_path.name)
            return
        
        # Calculate bounds from complete geometry
        bounds = self._calculate_complete_bounds(feature_set)
        
        # Set up the plot
        margin = max(VISUALIZATION_MARGIN_BASE_SMALL, (bounds["max_x"] - bounds["min_x"]) * VISUALIZATION_MARGIN_PERCENTAGE)
        ax.set_xlim(bounds["min_x"] - margin, bounds["max_x"] + margin)
        ax.set_ylim(bounds["min_y"] - margin, bounds["max_y"] + margin)
        ax.set_aspect('equal')
        ax.grid(True, alpha=VISUALIZATION_GRID_ALPHA)
        
        # Plot complete DXF geometry
        if feature_set.reference_geometry:
            self._plot_reference_geometry(ax, feature_set.reference_geometry)
            
        # Plot reconstructed geometry
        if feature_set.reconstructed_geometry:
            self._plot_reconstructed_geometry(ax, feature_set.reconstructed_geometry)
        
        # Plot expected features
        self._plot_expected_features(ax, feature_set.features, show_labels=False)
        
        # Set title
        plot_title = title or feature_set.source_dxf_path.name
        plot_title += f"\n{feature_set.total_feature_count} features"
        ax.set_title(plot_title, fontsize=VISUALIZATION_DEFAULT_FONT_SIZE)

    def _plot_single_feature_set(self, ax, feature_set: ExpectedFeatureSet, title: str = None):
        """Legacy method - redirects to complete feature set plotting."""
        return self._plot_single_complete_feature_set(ax, feature_set, title)
        
    # Alias for backwards compatibility
    def _add_comprehensive_legend(self, ax):
        """Comprehensive legend - redirects to main legend method."""
        return self._add_legend(ax)
    
    def _calculate_complete_bounds(self, feature_set: ExpectedFeatureSet) -> Dict[str, float]:
        """Calculate plotting bounds from COMPLETE DXF geometry, not just features."""
        
        x_coords = []
        y_coords = []
        
        # Include reference geometry bounds
        if feature_set.reference_geometry:
            for entity in feature_set.reference_geometry:
                entity_bounds = self._get_entity_bounds(entity)
                if entity_bounds:
                    x_coords.extend([entity_bounds["min_x"], entity_bounds["max_x"]])
                    y_coords.extend([entity_bounds["min_y"], entity_bounds["max_y"]])
        
        # Include reconstructed geometry bounds
        if (feature_set.reconstructed_geometry and 
            hasattr(feature_set.reconstructed_geometry, 'reconstructed_circles')):
            for circle in feature_set.reconstructed_geometry.reconstructed_circles:
                x_coords.extend([
                    circle.center.x - circle.radius,
                    circle.center.x + circle.radius
                ])
                y_coords.extend([
                    circle.center.y - circle.radius,
                    circle.center.y + circle.radius
                ])
        
        # Include feature bounds as well
        for feature in feature_set.features:
            x_coords.append(feature.center.x)
            y_coords.append(feature.center.y)
            
            if feature.radius:
                x_coords.extend([
                    feature.center.x - feature.radius,
                    feature.center.x + feature.radius
                ])
                y_coords.extend([
                    feature.center.y - feature.radius,
                    feature.center.y + feature.radius
                ])
        
        # Default bounds if no geometry found
        if not x_coords or not y_coords:
            return {"min_x": -VISUALIZATION_DEFAULT_BOUNDS, "max_x": VISUALIZATION_DEFAULT_BOUNDS, 
                   "min_y": -VISUALIZATION_DEFAULT_BOUNDS, "max_y": VISUALIZATION_DEFAULT_BOUNDS}
        
        return {
            "min_x": min(x_coords),
            "max_x": max(x_coords),
            "min_y": min(y_coords),
            "max_y": max(y_coords)
        }
        
    def _get_entity_bounds(self, entity: NormalizedEntity) -> Optional[Dict[str, float]]:
        """Get bounding box for a normalized entity."""
        
        if entity.source_entity.entity_type == EntityType.LINE:
            if entity.source_entity.start_point and entity.source_entity.end_point:
                return {
                    "min_x": min(entity.source_entity.start_point.x, entity.source_entity.end_point.x),
                    "max_x": max(entity.source_entity.start_point.x, entity.source_entity.end_point.x),
                    "min_y": min(entity.source_entity.start_point.y, entity.source_entity.end_point.y),
                    "max_y": max(entity.source_entity.start_point.y, entity.source_entity.end_point.y)
                }
                
        elif entity.source_entity.entity_type in [EntityType.CIRCLE, EntityType.ARC]:
            if entity.source_entity.center and entity.source_entity.radius:
                return {
                    "min_x": entity.source_entity.center.x - entity.source_entity.radius,
                    "max_x": entity.source_entity.center.x + entity.source_entity.radius,
                    "min_y": entity.source_entity.center.y - entity.source_entity.radius,
                    "max_y": entity.source_entity.center.y + entity.source_entity.radius
                }
        
        return None
        
    def _plot_reference_geometry(self, ax, reference_entities: List[NormalizedEntity]):
        """Plot complete DXF reference geometry as background."""
        
        logger.debug(f"Plotting {len(reference_entities)} reference entities")
        
        for entity in reference_entities:
            self._plot_dxf_entity(ax, entity)
            
    def _plot_dxf_entity(self, ax, entity: NormalizedEntity):
        """Plot a single DXF entity with appropriate geometry."""
        
        entity_type = entity.source_entity.entity_type
        
        if entity_type == EntityType.LINE:
            self._plot_line_entity(ax, entity)
        elif entity_type == EntityType.CIRCLE:
            self._plot_circle_entity(ax, entity)
        elif entity_type == EntityType.ARC:
            self._plot_arc_entity(ax, entity)
            
    def _plot_line_entity(self, ax, entity: NormalizedEntity):
        """Plot a LINE entity."""
        
        if not (entity.source_entity.start_point and entity.source_entity.end_point):
            return
            
        start = entity.source_entity.start_point
        end = entity.source_entity.end_point
        
        ax.plot([start.x, end.x], [start.y, end.y], 
               color=self.dxf_colors["line"], linewidth=VISUALIZATION_LINEWIDTH_DXF, alpha=VISUALIZATION_ALPHA_DXF_GEOMETRY)
               
    def _plot_circle_entity(self, ax, entity: NormalizedEntity):
        """Plot a CIRCLE entity."""
        
        if not (entity.source_entity.center and entity.source_entity.radius):
            return
            
        circle = Circle(
            (entity.source_entity.center.x, entity.source_entity.center.y),
            entity.source_entity.radius,
            fill=False,
            edgecolor=self.dxf_colors["circle"],
            linewidth=VISUALIZATION_LINEWIDTH_DXF,
            alpha=VISUALIZATION_ALPHA_DXF_GEOMETRY
        )
        ax.add_patch(circle)
        
    def _plot_arc_entity(self, ax, entity: NormalizedEntity):
        """Plot an ARC entity with correct angular span."""
        
        if not (entity.source_entity.center and entity.source_entity.radius and
                entity.source_entity.start_angle is not None and 
                entity.source_entity.end_angle is not None):
            return
            
        center = entity.source_entity.center
        radius = entity.source_entity.radius
        start_angle = entity.source_entity.start_angle
        end_angle = entity.source_entity.end_angle
        
        # Handle angle wraparound
        if end_angle < start_angle:
            end_angle += VISUALIZATION_ARC_ANGLE_FULL_CIRCLE
        
        arc = Arc(
            (center.x, center.y),
            VISUALIZATION_RADIUS_ARC_MULTIPLIER * radius,  # width
            VISUALIZATION_RADIUS_ARC_MULTIPLIER * radius,  # height  
            angle=0,
            theta1=start_angle,
            theta2=end_angle,
            color=self.dxf_colors["arc"],
            linewidth=VISUALIZATION_LINEWIDTH_DXF,
            alpha=VISUALIZATION_ALPHA_DXF_GEOMETRY
        )
        ax.add_patch(arc)

    def _plot_expected_features(self, ax, features: List[ExpectedFeature], show_labels: bool = True):
        """Plot expected features as prominent overlays."""
        
        logger.debug(f"Plotting {len(features)} expected features")
        
        for feature in features:
            self._plot_single_feature(ax, feature, show_labels)
            
    def _plot_single_feature(self, ax, feature: ExpectedFeature, show_labels: bool):
        """Plot a single expected feature with appropriate highlighting."""
        
        # Determine feature style
        if feature.feature_type == FeatureType.CIRCLE:
            color = self.feature_colors["circle"]
            linestyle = '-'
            linewidth = 3
        elif feature.feature_type == FeatureType.THROUGH_HOLE:
            # Check if it's a square hole
            shape = feature.geometric_properties.get('shape', '')
            if 'square' in shape.lower() or 'rectangle' in shape.lower():
                color = self.feature_colors["square_hole"]
                linestyle = '--'
                linewidth = 3
                self._plot_square_hole_feature(ax, feature, color, linestyle, linewidth)
                return
            else:
                color = self.feature_colors["through_hole"]
                linestyle = '--'
                linewidth = 3
        
        # Adjust alpha based on confidence
        alpha = VISUALIZATION_ALPHA_FEATURE_BASE + (feature.confidence * VISUALIZATION_ALPHA_FEATURE_CONFIDENCE_FACTOR)  # Range: 0.6 to 1.0
        
        # Draw feature circle
        if feature.radius:
            circle = Circle(
                (feature.center.x, feature.center.y),
                feature.radius,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha
            )
            ax.add_patch(circle)
        
        # Mark center point prominently
        ax.plot(feature.center.x, feature.center.y, 
               'o', color=color, markersize=6, alpha=alpha, markeredgewidth=2)
        
        # Add label if requested
        if show_labels:
            self._add_feature_label(ax, feature, color)
    
    def _plot_square_hole_feature(self, ax, feature: ExpectedFeature, color: str, linestyle: str, linewidth: int):
        """Plot a square/rectangular hole feature."""
        
        # Get dimensions from geometric properties
        width = feature.geometric_properties.get('width', VISUALIZATION_DEFAULT_FEATURE_WIDTH)
        height = feature.geometric_properties.get('height', VISUALIZATION_DEFAULT_FEATURE_HEIGHT)
        
        # Calculate rectangle position (center-based)
        rect_x = feature.center.x - width / 2
        rect_y = feature.center.y - height / 2
        
        alpha = VISUALIZATION_ALPHA_FEATURE_BASE + (feature.confidence * VISUALIZATION_ALPHA_FEATURE_CONFIDENCE_FACTOR)
        
        # Draw rectangle
        rectangle = Rectangle(
            (rect_x, rect_y),
            width,
            height,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha
        )
        ax.add_patch(rectangle)
        
        # Mark center point
        ax.plot(feature.center.x, feature.center.y,
               's', color=color, markersize=6, alpha=alpha, markeredgewidth=2)
               
    def _add_feature_label(self, ax, feature: ExpectedFeature, color: str):
        """Add a label to a feature."""
        
        # Create label text
        label_text = f"{feature.feature_id}\n{feature.feature_type.value}"
        
        if feature.radius:
            label_text += f"\nR={feature.radius:.1f}"
        else:
            width = feature.geometric_properties.get('width', 0)
            height = feature.geometric_properties.get('height', 0)
            if width and height:
                label_text += f"\n{width:.1f}×{height:.1f}"
                
        label_text += f"\nC={feature.confidence:.2f}"
        
        # Check if this is a central hole
        is_central = feature.detection_evidence.get('evidence_details', {}).get('central_position', {}).get('is_central', False)
        if is_central:
            label_text += "\n[CENTRAL]"
        
        # Position label with appropriate offset
        offset_distance = max(feature.radius if feature.radius else VISUALIZATION_LABEL_OFFSET_BASE, VISUALIZATION_LABEL_OFFSET_MIN)
        offset_x = offset_distance * VISUALIZATION_LABEL_OFFSET_FACTOR_X
        offset_y = offset_distance * VISUALIZATION_LABEL_OFFSET_FACTOR_Y
        
        ax.annotate(label_text,
                  (feature.center.x, feature.center.y),
                  xytext=(feature.center.x + offset_x, feature.center.y + offset_y),
                  fontsize=VISUALIZATION_LABEL_FONT_SIZE,
                  ha='left', va='bottom',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=VISUALIZATION_BBOX_ALPHA, edgecolor=color),
                  arrowprops=dict(arrowstyle='->', color=color, alpha=VISUALIZATION_ARROW_ALPHA, lw=1))

    def _add_legend(self, ax):
        """Add comprehensive legend explaining all visualization elements."""
        
        legend_elements = [
            # DXF Reference Geometry
            plt.Line2D([0], [0], color=self.dxf_colors["line"], linewidth=VISUALIZATION_LINEWIDTH_DXF, alpha=VISUALIZATION_ALPHA_DXF_GEOMETRY,
                      label='DXF Reference Geometry'),
            plt.Line2D([0], [0], color=self.dxf_colors["reconstructed"], linewidth=VISUALIZATION_LINEWIDTH_RECONSTRUCTED, 
                      linestyle='--', alpha=VISUALIZATION_ALPHA_RECONSTRUCTED_GEOMETRY, label='Reconstructed Geometry'),
            
            # Expected Features
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.feature_colors["circle"],
                      markersize=8, label='Expected Circle', linestyle='-', linewidth=3, 
                      markeredgecolor=self.feature_colors["circle"]),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.feature_colors["through_hole"],
                      markersize=8, label='Expected Circular Hole', linestyle='--', linewidth=3,
                      markeredgecolor=self.feature_colors["through_hole"]),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=self.feature_colors["square_hole"],
                      markersize=8, label='Expected Square Hole', linestyle='--', linewidth=3,
                      markeredgecolor=self.feature_colors["square_hole"]),
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0),
                 framealpha=VISUALIZATION_LEGEND_ALPHA)
    
                 
    def _plot_reconstructed_geometry(self, ax, reconstructed_geometry):
        """Plot reconstructed geometry structures."""
        
        if not hasattr(reconstructed_geometry, 'reconstructed_circles'):
            return
            
        logger.debug(f"Plotting {len(reconstructed_geometry.reconstructed_circles)} reconstructed circles")
        
        for circle in reconstructed_geometry.reconstructed_circles:
            # Plot reconstructed circles with slightly different style
            circle_patch = Circle(
                (circle.center.x, circle.center.y),
                circle.radius,
                fill=False,
                edgecolor=self.dxf_colors["reconstructed"],
                linewidth=VISUALIZATION_LINEWIDTH_RECONSTRUCTED,
                alpha=VISUALIZATION_ALPHA_RECONSTRUCTED_GEOMETRY,
                linestyle='--'
            )
            ax.add_patch(circle_patch)

    def _add_metadata_text(self, ax, feature_set: ExpectedFeatureSet):
        """Add enhanced metadata text box."""
        
        # Create metadata text
        metadata_lines = [
            f"Source: {feature_set.source_dxf_path.name}",
            f"Units: {feature_set.dxf_units}",
            f"DXF entities: {feature_set.normalized_entity_count}",
            f"Reconstructed: {feature_set.reconstructed_geometry_count}",
            f"Avg confidence: {feature_set.average_confidence:.3f}",
            f"Extraction: {feature_set.extraction_timestamp.split('T')[0]}"
        ]
        
        metadata_text = "\n".join(metadata_lines)
        
        # Add text box
        ax.text(VISUALIZATION_TEXT_CORNER_OFFSET, VISUALIZATION_TEXT_CORNER_OFFSET, metadata_text,
               transform=ax.transAxes,
               fontsize=VISUALIZATION_LABEL_FONT_SIZE,
               verticalalignment='bottom',
               bbox=dict(boxstyle=f'round,pad={VISUALIZATION_BBOX_PAD_LARGE}', facecolor='lightgray', alpha=VISUALIZATION_BBOX_ALPHA))


def visualize_expected_features(feature_set: ExpectedFeatureSet, 
                              output_path: Optional[Path] = None,
                              show_entities: bool = True,
                              show_labels: bool = True) -> Path:
    """
    Convenience function to visualize expected features.
    
    Args:
        feature_set: The feature set to visualize
        output_path: Optional output file path
        show_entities: Whether to show underlying DXF entities
        show_labels: Whether to show feature labels
        
    Returns:
        Path to saved visualization
    """
    visualizer = ExpectedFeatureVisualizer()
    return visualizer.visualize_feature_set(
        feature_set, output_path, show_entities, show_labels
    )