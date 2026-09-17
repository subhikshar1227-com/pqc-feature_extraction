"""
DXF Processing Module

Provides parsing, normalization, and geometric analysis of DXF files.

Public API:
    parse_dxf(dxf_path: Path) -> List[DxfEntity]
    normalize_geometry(entities: List[DxfEntity]) -> List[NormalizedEntity]
    analyze_relationships(entities: List[NormalizedEntity]) -> GeometricRelationships
    reconstruct_geometry(entities, relationships) -> List[ReconstructedGeometry]
"""

from .parser import parse_dxf
from .entity_models import DxfEntity, NormalizedEntity, EntityType
from .geometry_normalizer import normalize_geometry
from .geometry_reconstruction import reconstruct_geometry, analyze_relationships

__all__ = [
    "parse_dxf",
    "normalize_geometry", 
    "analyze_relationships",
    "reconstruct_geometry",
    "DxfEntity",
    "NormalizedEntity", 
    "EntityType"
]