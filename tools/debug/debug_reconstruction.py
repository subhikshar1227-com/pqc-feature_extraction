from feature_extraction.dxf.parser import parse_dxf
from feature_extraction.dxf.geometry_normalizer import normalize_geometry
from feature_extraction.dxf.geometry_reconstruction import analyze_relationships, reconstruct_geometry
from pathlib import Path

entities = parse_dxf(Path('dxf/cad_box_front.dxf'))
normalized = normalize_geometry(entities)
relationships = analyze_relationships(normalized)
reconstructed = reconstruct_geometry(normalized, relationships)

print(f'Total entities: {len(entities)}')
print(f'Normalized entities: {len(normalized)}')
print(f'Relationships: {len(relationships.relationships)}')
print(f'Reconstructed circles: {len(reconstructed.reconstructed_circles)}')
print(f'Arc groups analyzed: {reconstructed.arc_groups_analyzed}')
print(f'Arc groups rejected: {reconstructed.arc_groups_rejected}')
print(f'Rejection reasons: {reconstructed.rejection_reasons}')

if reconstructed.reconstructed_circles:
    for i, circle in enumerate(reconstructed.reconstructed_circles):
        print(f'Circle {i+1}: center=({circle.center.x:.2f}, {circle.center.y:.2f}), radius={circle.radius:.2f}, confidence={circle.confidence:.3f}')
        print(f'  Coverage: {circle.coverage_fraction:.2f}, Source entities: {len(circle.source_entities)}')

# Let's also check what arc relationships exist
arc_compatible = [rel for rel in relationships.relationships if rel.relationship_type == "arc_compatible"]
print(f'\nArc compatible relationships: {len(arc_compatible)}')

for i, rel in enumerate(arc_compatible[:5]):  # Show first 5
    print(f'  {i+1}. {rel.entity1_id} <-> {rel.entity2_id}, confidence={rel.confidence:.3f}')