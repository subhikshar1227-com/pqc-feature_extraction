"""
DXF Parser Implementation

Parses DXF files using ezdxf and converts to clean internal representations.
"""

import logging
from pathlib import Path
from typing import List, Optional
import ezdxf
from ezdxf.document import Drawing

from .entity_models import DxfEntity, EntityType, Point2D
from ..config import (
    GEOMETRIC_COORDINATE_PRECISION, GEOMETRIC_SIZE_PRECISION, GEOMETRIC_ANGLE_PRECISION
)

logger = logging.getLogger(__name__)


class DxfParsingError(Exception):
    """Raised when DXF parsing fails."""
    pass


def parse_dxf(dxf_path: Path) -> List[DxfEntity]:
    """
    Parse a DXF file and extract geometric entities.
    
    Args:
        dxf_path: Path to the DXF file
        
    Returns:
        List of DxfEntity objects representing the geometric content
        
    Raises:
        DxfParsingError: If the DXF cannot be parsed or is invalid
    """
    if not dxf_path.exists():
        raise DxfParsingError(f"DXF file not found: {dxf_path}")
    
    if not dxf_path.is_file():
        raise DxfParsingError(f"Path is not a file: {dxf_path}")
    
    try:
        logger.debug(f"Parsing DXF file: {dxf_path}")
        doc = ezdxf.readfile(str(dxf_path))
        
        # Get the model space (main drawing area)
        msp = doc.modelspace()
        
        entities = []
        entity_counter = 0
        
        for ezdxf_entity in msp:
            entity_counter += 1
            
            try:
                dxf_entity = _convert_ezdxf_entity(ezdxf_entity, entity_counter)
                if dxf_entity is not None:
                    entities.append(dxf_entity)
                    
            except Exception as e:
                logger.warning(f"Failed to convert entity {entity_counter} "
                             f"(type: {ezdxf_entity.dxftype()}): {e}")
                continue
        
        logger.info(f"Successfully parsed {len(entities)} entities from {dxf_path.name}")
        
        # Log entity type summary
        entity_types = {}
        for entity in entities:
            entity_types[entity.entity_type.value] = entity_types.get(entity.entity_type.value, 0) + 1
        
        logger.debug(f"Entity type summary: {entity_types}")
        
        # ENTITY ORDER INDEPENDENCE FIX:
        # Sort entities by geometric properties to ensure deterministic processing
        # regardless of DXF entity order
        entities = _sort_entities_deterministically(entities)
        
        return entities
        
    except ezdxf.DXFError as e:
        raise DxfParsingError(f"EzDXF parsing error for {dxf_path}: {e}")
    except Exception as e:
        raise DxfParsingError(f"Unexpected error parsing {dxf_path}: {e}")


def _convert_ezdxf_entity(ezdxf_entity, entity_id: int) -> Optional[DxfEntity]:
    """
    Convert an ezdxf entity to our internal DxfEntity representation.
    
    Args:
        ezdxf_entity: Raw ezdxf entity object
        entity_id: Sequential entity identifier
        
    Returns:
        DxfEntity object or None if entity type not supported
    """
    entity_type_name = ezdxf_entity.dxftype()
    handle = getattr(ezdxf_entity.dxf, 'handle', f'synthetic_{entity_id}')
    layer = getattr(ezdxf_entity.dxf, 'layer', 'default')
    
    # Generate unique entity ID
    entity_id_str = f"{entity_type_name}_{handle}_{entity_id}"
    
    if entity_type_name == 'CIRCLE':
        return _convert_circle(ezdxf_entity, entity_id_str, layer, handle)
        
    elif entity_type_name == 'ARC':
        return _convert_arc(ezdxf_entity, entity_id_str, layer, handle)
        
    elif entity_type_name == 'LINE':
        return _convert_line(ezdxf_entity, entity_id_str, layer, handle)
        
    elif entity_type_name == 'LWPOLYLINE':
        return _convert_lwpolyline(ezdxf_entity, entity_id_str, layer, handle)
        
    else:
        # Unsupported entity type - log but don't fail
        logger.debug(f"Skipping unsupported entity type: {entity_type_name}")
        return None


def _convert_circle(ezdxf_entity, entity_id: str, layer: str, handle: str) -> DxfEntity:
    """Convert ezdxf CIRCLE entity."""
    try:
        center_vec = ezdxf_entity.dxf.center
        radius = ezdxf_entity.dxf.radius
        
        return DxfEntity(
            entity_id=entity_id,
            entity_type=EntityType.CIRCLE,
            layer=layer,
            handle=handle,
            center=Point2D(center_vec.x, center_vec.y),
            radius=radius
        )
    except AttributeError as e:
        raise ValueError(f"Invalid CIRCLE entity: missing required attribute: {e}")


def _convert_arc(ezdxf_entity, entity_id: str, layer: str, handle: str) -> DxfEntity:
    """Convert ezdxf ARC entity.""" 
    try:
        center_vec = ezdxf_entity.dxf.center
        radius = ezdxf_entity.dxf.radius
        start_angle = ezdxf_entity.dxf.start_angle
        end_angle = ezdxf_entity.dxf.end_angle
        
        return DxfEntity(
            entity_id=entity_id,
            entity_type=EntityType.ARC,
            layer=layer,
            handle=handle,
            center=Point2D(center_vec.x, center_vec.y),
            radius=radius,
            start_angle=start_angle,
            end_angle=end_angle
        )
    except AttributeError as e:
        raise ValueError(f"Invalid ARC entity: missing required attribute: {e}")


def _convert_line(ezdxf_entity, entity_id: str, layer: str, handle: str) -> DxfEntity:
    """Convert ezdxf LINE entity."""
    try:
        start_vec = ezdxf_entity.dxf.start
        end_vec = ezdxf_entity.dxf.end
        
        return DxfEntity(
            entity_id=entity_id,
            entity_type=EntityType.LINE,
            layer=layer,
            handle=handle,
            start_point=Point2D(start_vec.x, start_vec.y),
            end_point=Point2D(end_vec.x, end_vec.y)
        )
    except AttributeError as e:
        raise ValueError(f"Invalid LINE entity: missing required attribute: {e}")


def _convert_lwpolyline(ezdxf_entity, entity_id: str, layer: str, handle: str) -> DxfEntity:
    """Convert ezdxf LWPOLYLINE entity."""
    try:
        # Get points from the polyline
        points = []
        for point in ezdxf_entity.get_points():
            points.append(Point2D(point[0], point[1]))
        
        # Check if polyline is closed
        is_closed = ezdxf_entity.closed
        
        return DxfEntity(
            entity_id=entity_id,
            entity_type=EntityType.LWPOLYLINE,
            layer=layer, 
            handle=handle,
            points=points,
            is_closed=is_closed
        )
    except Exception as e:
        raise ValueError(f"Invalid LWPOLYLINE entity: {e}")


def get_dxf_units(dxf_path: Path) -> str:
    """
    Extract units information from DXF file.
    
    Args:
        dxf_path: Path to DXF file
        
    Returns:
        String describing the units (e.g., "mm", "inches", "units")
    """
    try:
        doc = ezdxf.readfile(str(dxf_path))
        units_code = doc.units
        
        # Map ezdxf units codes to readable names
        units_map = {
            0: "unitless",
            1: "inches", 
            2: "feet",
            3: "miles",
            4: "mm",
            5: "cm", 
            6: "m",
            7: "km"
        }
        
        return units_map.get(units_code, f"unknown_units_{units_code}")
        
    except Exception as e:
        logger.warning(f"Could not determine units for {dxf_path}: {e}")
        return "unknown_units"


def _sort_entities_deterministically(entities: List[DxfEntity]) -> List[DxfEntity]:
    """
    Sort entities by geometric properties to ensure deterministic processing
    regardless of DXF file entity order.
    
    This fixes entity-order dependency issues in feature extraction.
    
    Args:
        entities: List of parsed DXF entities
        
    Returns:
        Sorted list of entities with deterministic order
    """
    def geometric_sort_key(entity: DxfEntity):
        """Create a deterministic sort key based on geometric properties."""
        
        # Primary sort: entity type (for consistent grouping)
        type_priority = {
            EntityType.CIRCLE: 1,
            EntityType.ARC: 2, 
            EntityType.LINE: 3,
            EntityType.LWPOLYLINE: 4
        }
        
        # Secondary sort: geometric center (for spatial ordering)
        center_x = 0.0
        center_y = 0.0
        
        if entity.center:
            center_x = round(entity.center.x, GEOMETRIC_COORDINATE_PRECISION)  # Configurable precision for coordinate normalization
            center_y = round(entity.center.y, GEOMETRIC_COORDINATE_PRECISION)
        elif entity.start_point:
            # For lines, use start point
            center_x = round(entity.start_point.x, GEOMETRIC_COORDINATE_PRECISION)
            center_y = round(entity.start_point.y, GEOMETRIC_COORDINATE_PRECISION)
        
        # Tertiary sort: geometric size (radius for circles/arcs, length for lines)
        size = 0.0
        if entity.radius is not None:
            size = round(entity.radius, GEOMETRIC_SIZE_PRECISION)
        elif entity.start_point and entity.end_point:
            # For lines, use length
            dx = entity.end_point.x - entity.start_point.x
            dy = entity.end_point.y - entity.start_point.y
            size = round((dx*dx + dy*dy)**0.5, GEOMETRIC_SIZE_PRECISION)
            
        # Quaternary sort: angles for arcs (to distinguish between arcs at same location)
        start_angle = 0.0
        end_angle = 0.0
        if entity.start_angle is not None:
            start_angle = round(entity.start_angle, GEOMETRIC_ANGLE_PRECISION)
        if entity.end_angle is not None:
            end_angle = round(entity.end_angle, GEOMETRIC_ANGLE_PRECISION)
        
        # Final sort: layer and handle for complete determinism
        layer = entity.layer or ""
        handle = entity.handle or ""
        
        return (
            type_priority.get(entity.entity_type, 999),  # Entity type priority
            center_x,                                     # X coordinate
            center_y,                                     # Y coordinate  
            size,                                         # Geometric size
            start_angle,                                  # Start angle (arcs)
            end_angle,                                    # End angle (arcs)
            layer,                                        # Layer name
            handle                                        # Entity handle
        )
    
    # Sort entities using the geometric key
    sorted_entities = sorted(entities, key=geometric_sort_key)
    
    # Regenerate entity IDs based on sorted order to ensure consistency
    for i, entity in enumerate(sorted_entities, 1):
        # Create new deterministic ID based on sorted position
        entity_type_name = entity.entity_type.value
        handle = entity.handle
        entity.entity_id = f"{entity_type_name}_{handle}_{i}"
    
    logger.debug(f"Sorted {len(entities)} entities deterministically by geometric properties")
    
    return sorted_entities