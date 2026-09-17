from feature_extraction.dxf.parser import parse_dxf
from feature_extraction.dxf.geometry_normalizer import normalize_geometry
from feature_extraction.expected.through_hole_detector import ThroughHoleDetector
from feature_extraction.dxf.geometry_reconstruction import analyze_relationships, reconstruct_geometry
from pathlib import Path

# Test central hole detection for c-bp.dxf
entities = parse_dxf(Path('dxf/c-bp.dxf'))
normalized = normalize_geometry(entities)
relationships = analyze_relationships(normalized)
reconstructed = reconstruct_geometry(normalized, relationships)

# Find the circle at (152.16, 146.84)
target_circle = None
for entity in normalized:
    if (entity.source_entity.entity_type.name == 'CIRCLE' and
        entity.source_entity.center and 
        abs(entity.source_entity.center.x - 152.16) < 0.1 and
        abs(entity.source_entity.center.y - 146.84) < 0.1):
        target_circle = entity
        break

if target_circle:
    print(f'=== CENTRAL HOLE ANALYSIS FOR c-bp.dxf ===')
    print(f'Target circle: center=({target_circle.source_entity.center.x:.2f}, {target_circle.source_entity.center.y:.2f}), '
          f'radius={target_circle.source_entity.radius:.2f}')
    
    # Test the through hole detector on this specific circle
    detector = ThroughHoleDetector()
    
    # Call the internal analysis method
    geometric_context = detector._analyze_geometric_context(target_circle, normalized)
    print(f'Geometric context: {geometric_context}')
    
    evidence = detector._analyze_comprehensive_hole_evidence(target_circle, normalized, relationships)
    print(f'Evidence analysis:')
    print(f'  Total confidence: {evidence["total_confidence"]:.3f}')
    print(f'  Geometric closure: {evidence["geometric_closure"]:.3f}')
    print(f'  Concentric nesting: {evidence["concentric_nesting"]:.3f}')
    print(f'  Pattern regularity: {evidence["pattern_regularity"]:.3f}')
    print(f'  Size appropriateness: {evidence["size_appropriateness"]:.3f}')
    print(f'  Layer context: {evidence["layer_context"]:.3f}')
    
    central_details = evidence["evidence_details"].get("central_position", {})
    print(f'  Central position analysis:')
    print(f'    Is central: {central_details.get("is_central", False)}')
    print(f'    Evidence score: {central_details.get("evidence_score", 0.0):.3f}')
    print(f'    Analysis type: {central_details.get("analysis_type", "unknown")}')
    
    print(f'  Required confidence threshold: {detector.min_confidence}')
    print(f'  Would be classified as hole: {evidence["total_confidence"] >= detector.min_confidence}')
    
else:
    print('Target circle not found!')