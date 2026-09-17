from feature_extraction.dxf.parser import parse_dxf
from feature_extraction.dxf.geometry_normalizer import normalize_geometry
from feature_extraction.dxf.geometry_reconstruction import analyze_relationships, reconstruct_geometry
from pathlib import Path

# Debug box_front specifically
entities = parse_dxf(Path('dxf/cad_box_front.dxf'))
normalized = normalize_geometry(entities)
relationships = analyze_relationships(normalized)
reconstructed = reconstruct_geometry(normalized, relationships)

print(f'=== BOX FRONT DETAILED DEBUG ===')
print(f'Total entities: {len(entities)}')
print(f'Normalized entities: {len(normalized)}')

# Check arc entities specifically
arc_entities = [e for e in normalized if e.source_entity.entity_type.name == 'ARC']
print(f'Arc entities: {len(arc_entities)}')

for i, arc in enumerate(arc_entities[:10]):  # Show first 10 arcs
    center = arc.source_entity.center
    radius = arc.source_entity.radius
    start = arc.source_entity.start_angle
    end = arc.source_entity.end_angle
    span = abs(end - start) if start is not None and end is not None else None
    
    print(f'  Arc {i+1} ({arc.source_entity.entity_id}): '
          f'center=({center.x:.1f},{center.y:.1f}), '
          f'radius={radius:.1f}, '
          f'angles={start:.1f}°-{end:.1f}° (span={span:.1f}°)')

print(f'\nRelationships: {len(relationships.relationships)}')
arc_compatible = [rel for rel in relationships.relationships if rel.relationship_type == "arc_compatible"]
print(f'Arc compatible relationships: {len(arc_compatible)}')

for i, rel in enumerate(arc_compatible[:10]):
    print(f'  {i+1}. {rel.entity1_id} <-> {rel.entity2_id}, confidence={rel.confidence:.3f}')

print(f'\nReconstruction results:')
print(f'Reconstructed circles: {len(reconstructed.reconstructed_circles)}')
print(f'Arc groups analyzed: {reconstructed.arc_groups_analyzed}')
print(f'Arc groups rejected: {reconstructed.arc_groups_rejected}')
print(f'Rejection reasons: {reconstructed.rejection_reasons}')

if reconstructed.reconstructed_circles:
    for i, circle in enumerate(reconstructed.reconstructed_circles):
        print(f'Circle {i+1}: center=({circle.center.x:.2f}, {circle.center.y:.2f}), '
              f'radius={circle.radius:.2f}, confidence={circle.confidence:.3f}')
        print(f'  Coverage: {circle.coverage_fraction:.2f}, '
              f'Source entities: {len(circle.source_entities)}, '
              f'Gaps: {circle.gaps}, Overlaps: {circle.overlaps}')

# Now test the full feature extraction
print(f'\n=== FULL FEATURE EXTRACTION TEST ===')
from feature_extraction.expected_feature_extractor import extract_expected_features

feature_set = extract_expected_features(Path('dxf/cad_box_front.dxf'))
print(f'Total features: {feature_set.total_feature_count}')
print(f'Circles: {feature_set.circle_count}, Through holes: {feature_set.through_hole_count}')

for i, feature in enumerate(feature_set.features):
    feature_type = feature.feature_type.value
    center = feature.center
    confidence = feature.confidence
    
    if hasattr(feature, 'radius') and feature.radius is not None:
        size_info = f'radius={feature.radius:.2f}'
    else:
        width = feature.geometric_properties.get('width', 0)
        height = feature.geometric_properties.get('height', 0)
        shape = feature.geometric_properties.get('shape', 'unknown')
        size_info = f'{shape} {width:.1f}x{height:.1f}' if width and height else 'no_size'
    
    print(f'  {i+1}. {feature_type}: center=({center.x:.2f}, {center.y:.2f}), {size_info}, conf={confidence:.3f}')