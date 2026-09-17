"""
DXF Resolution Module

Provides centralized blueprint-to-DXF mapping for the Peenya Project MSME
quality inspection system.

Public API:
    resolve_dxf(blueprint_name: str) -> Path
    get_blueprint_dxf_mapping() -> dict[str, str]
"""

from .resolver import resolve_dxf, get_blueprint_dxf_mapping, DXFResolutionError

__all__ = ["resolve_dxf", "get_blueprint_dxf_mapping", "DXFResolutionError"]
__version__ = "0.1.0"