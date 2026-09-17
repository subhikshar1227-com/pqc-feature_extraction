import sys
sys.path.append('../../')
from feature_extraction.expected_feature_extractor import extract_expected_features
from pathlib import Path

print('=== C-BP.DXF DETAILED DEBUG ===')
feature_set = extract_expected_features(Path('../../data/dxf/c-bp.dxf'))
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
    
    is_central = feature.detection_evidence.get('evidence_details', {}).get('central_position', {}).get('is_central', False)
    source_type = feature.source_type
    
    print(f'  {i+1}. {feature_type}: center=({center.x:.2f}, {center.y:.2f}), {size_info}, conf={confidence:.3f}, central={is_central}, source={source_type}')

print('\n=== EXPECTED TARGET ===')
print('c-bp.dxf target: 4 circles + 1 central circular hole')
print('Current result: 5 circles + 0 holes')
print('Issue: Need to identify 1 circle as central hole and filter out any main body circles')