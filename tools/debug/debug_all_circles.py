from feature_extraction.dxf.parser import parse_dxf
from feature_extraction.dxf.geometry_normalizer import normalize_geometry
from pathlib import Path

# Find all circles at (152.16, 146.84) in c-bp.dxf
entities = parse_dxf(Path('dxf/c-bp.dxf'))
normalized = normalize_geometry(entities)

print('=== ALL CIRCLES IN c-bp.dxf ===')
circles_at_target = []

for entity in normalized:
    if entity.source_entity.entity_type.name == 'CIRCLE':
        center = entity.source_entity.center
        radius = entity.source_entity.radius
        print(f'Circle {entity.source_entity.entity_id}: center=({center.x:.2f}, {center.y:.2f}), radius={radius:.2f}')
        
        # Check if this is at the target location
        if (abs(center.x - 152.16) < 0.1 and abs(center.y - 146.84) < 0.1):
            circles_at_target.append(entity)
            print(f'  *** TARGET LOCATION CIRCLE ***')

print(f'\nFound {len(circles_at_target)} circles at target location (152.16, 146.84)')
for i, circle in enumerate(circles_at_target):
    print(f'  {i+1}. {circle.source_entity.entity_id}: radius={circle.source_entity.radius:.2f}')

# Also check what the current feature extractor gives us
print('\n=== CURRENT FEATURE EXTRACTION RESULTS ===')
from feature_extraction.expected_feature_extractor import extract_expected_features
feature_set = extract_expected_features(Path('dxf/c-bp.dxf'))

for i, feature in enumerate(feature_set.features):
    center = feature.center
    print(f'{i+1}. {feature.feature_type.value}: center=({center.x:.2f}, {center.y:.2f}), radius={getattr(feature, "radius", "N/A")}')