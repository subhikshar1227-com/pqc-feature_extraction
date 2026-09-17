"""
DXF Entity Data Models

Provides clean internal representations of DXF geometric entities,
independent of the underlying ezdxf library structures.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Any
import math

from ..config import FULL_CIRCLE_ARC_SPAN_TOLERANCE


class EntityType(Enum):
    """Supported DXF entity types."""
    CIRCLE = "CIRCLE"
    ARC = "ARC"
    LINE = "LINE" 
    LWPOLYLINE = "LWPOLYLINE"
    UNKNOWN = "UNKNOWN"


@dataclass
class Point2D:
    """2D point in DXF coordinate space (mm)."""
    x: float
    y: float
    
    def distance_to(self, other: 'Point2D') -> float:
        """Calculate Euclidean distance to another point."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def __iter__(self):
        """Allow unpacking as (x, y)."""
        yield self.x
        yield self.y


@dataclass  
class BoundingBox:
    """Axis-aligned bounding box."""
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    
    @property
    def width(self) -> float:
        return self.max_x - self.min_x
    
    @property 
    def height(self) -> float:
        return self.max_y - self.min_y
        
    @property
    def center(self) -> Point2D:
        return Point2D(
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2
        )
    
    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class DxfEntity:
    """
    Clean representation of a DXF entity with normalized geometry.
    
    This internal representation is independent of ezdxf structures
    and contains only the geometric information needed for feature extraction.
    """
    entity_id: str                    # Unique identifier
    entity_type: EntityType           # Type of geometric entity
    layer: str                        # DXF layer name
    handle: str                       # Original DXF handle
    
    # Geometric properties (entity-type dependent)
    center: Optional[Point2D] = None          # For CIRCLE, ARC
    radius: Optional[float] = None            # For CIRCLE, ARC  
    start_angle: Optional[float] = None       # For ARC (degrees)
    end_angle: Optional[float] = None         # For ARC (degrees)
    start_point: Optional[Point2D] = None     # For LINE
    end_point: Optional[Point2D] = None       # For LINE
    points: Optional[List[Point2D]] = None    # For LWPOLYLINE
    is_closed: Optional[bool] = None          # For LWPOLYLINE
    
    # Computed properties
    bounding_box: Optional[BoundingBox] = None
    
    def __post_init__(self):
        """Compute derived properties after initialization."""
        if self.bounding_box is None:
            self.bounding_box = self._compute_bounding_box()
    
    def _compute_bounding_box(self) -> BoundingBox:
        """Compute the axis-aligned bounding box for this entity."""
        if self.entity_type in [EntityType.CIRCLE, EntityType.ARC]:
            if self.center and self.radius:
                return BoundingBox(
                    self.center.x - self.radius,
                    self.center.y - self.radius, 
                    self.center.x + self.radius,
                    self.center.y + self.radius
                )
        elif self.entity_type == EntityType.LINE:
            if self.start_point and self.end_point:
                return BoundingBox(
                    min(self.start_point.x, self.end_point.x),
                    min(self.start_point.y, self.end_point.y),
                    max(self.start_point.x, self.end_point.x),
                    max(self.start_point.y, self.end_point.y)
                )
        elif self.entity_type == EntityType.LWPOLYLINE and self.points:
            xs = [p.x for p in self.points]
            ys = [p.y for p in self.points]
            return BoundingBox(min(xs), min(ys), max(xs), max(ys))
        
        # Fallback for unknown or incomplete geometry
        return BoundingBox(0, 0, 0, 0)
    
    @property
    def arc_span_degrees(self) -> Optional[float]:
        """Calculate the angular span of an arc in degrees."""
        if self.entity_type != EntityType.ARC or self.start_angle is None or self.end_angle is None:
            return None
        
        span = self.end_angle - self.start_angle
        if span < 0:
            span += 360
        return span
    
    @property
    def is_full_circle_arc(self) -> bool:
        """Check if this arc represents a complete circle."""
        if self.entity_type != EntityType.ARC:
            return False
        span = self.arc_span_degrees
        return span is not None and abs(span - 360) < FULL_CIRCLE_ARC_SPAN_TOLERANCE  # Configurable tolerance for full circle detection


@dataclass
class NormalizedEntity:
    """
    Normalized geometric entity after coordinate system normalization.
    
    This represents DXF entities in a consistent coordinate system with
    additional computed geometric properties for relationship analysis.
    """
    source_entity: DxfEntity
    
    # Normalized geometric properties  
    normalized_center: Optional[Point2D] = None
    normalized_radius: Optional[float] = None
    normalized_points: Optional[List[Point2D]] = None
    
    # Geometric analysis results
    geometric_signature: Optional[str] = None      # Hash of geometric properties
    contains_point: Optional[Point2D] = None       # Representative internal point
    perimeter_length: Optional[float] = None       # Calculated perimeter
    enclosed_area: Optional[float] = None          # Calculated area (if closed)
    
    def __post_init__(self):
        """Compute normalized and derived properties."""
        self._normalize_coordinates()
        self._compute_geometric_properties()
    
    def _normalize_coordinates(self):
        """Apply coordinate normalization (currently identity transform)."""
        # For Phase 1, we preserve DXF coordinates as-is
        # Future phases may apply coordinate transforms here
        if self.source_entity.center:
            self.normalized_center = Point2D(
                self.source_entity.center.x,
                self.source_entity.center.y
            )
        if self.source_entity.radius:
            self.normalized_radius = self.source_entity.radius
        if self.source_entity.points:
            self.normalized_points = [
                Point2D(p.x, p.y) for p in self.source_entity.points
            ]
    
    def _compute_geometric_properties(self):
        """Compute additional geometric properties for analysis."""
        entity = self.source_entity
        
        # Compute geometric signature for similarity comparison
        if entity.entity_type == EntityType.CIRCLE:
            self.geometric_signature = f"CIRCLE_R{entity.radius:.2f}"
            self.enclosed_area = math.pi * (entity.radius ** 2) if entity.radius else None
            self.perimeter_length = 2 * math.pi * entity.radius if entity.radius else None
            self.contains_point = self.normalized_center
            
        elif entity.entity_type == EntityType.ARC:
            span = entity.arc_span_degrees or 0
            self.geometric_signature = f"ARC_R{entity.radius:.2f}_S{span:.1f}"
            if entity.radius:
                self.perimeter_length = (span / 360) * 2 * math.pi * entity.radius
            
        elif entity.entity_type == EntityType.LINE:
            if entity.start_point and entity.end_point:
                length = entity.start_point.distance_to(entity.end_point)
                self.perimeter_length = length
                self.geometric_signature = f"LINE_L{length:.2f}"
                self.contains_point = Point2D(
                    (entity.start_point.x + entity.end_point.x) / 2,
                    (entity.start_point.y + entity.end_point.y) / 2
                )


@dataclass
class GeometricRelationship:
    """Represents a geometric relationship between two entities."""
    entity1_id: str
    entity2_id: str
    relationship_type: str                    # e.g., "concentric", "tangent", "connected"
    confidence: float                         # 0.0 to 1.0
    distance: Optional[float] = None          # Distance between entities
    angle: Optional[float] = None             # Angular relationship (degrees)
    overlap_factor: Optional[float] = None    # Amount of geometric overlap
    parameters: Optional[dict] = None         # Additional relationship-specific data


@dataclass
class GeometricRelationships:
    """Collection of relationships between entities with analysis metadata."""
    relationships: List[GeometricRelationship]
    entity_groups: List[List[str]]            # Groups of related entity IDs
    analysis_metadata: dict                   # Analysis statistics and parameters
    
    def get_relationships_for_entity(self, entity_id: str) -> List[GeometricRelationship]:
        """Get all relationships involving a specific entity."""
        return [
            rel for rel in self.relationships 
            if rel.entity1_id == entity_id or rel.entity2_id == entity_id
        ]