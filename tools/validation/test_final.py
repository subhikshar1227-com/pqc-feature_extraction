from feature_extraction.expected_feature_extractor import extract_expected_features
from pathlib import Path

# Test all four DXF files
files_to_test = [
    ('c-bp.dxf', 'c-bp'),
    ('c_tp.dxf', 'c_tp'), 
    ('cad_box_front.dxf', 'box_front'),
    ('cad_box_rear (1).dxf', 'box_rear')
]

print('=== PHASE 1 FINAL CORRECTION RESULTS ===')
print()

for dxf_file, name in files_to_test:
    print(f'--- {name} ({dxf_file}) ---')
    try:
        feature_set = extract_expected_features(Path(f'dxf/{dxf_file}'))
        print(f'Total features: {feature_set.total_feature_count}')
        print(f'Circles: {feature_set.circle_count}, Through holes: {feature_set.through_hole_count}')
        
        for i, feature in enumerate(feature_set.features[:10]):  # Show first 10
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
            
            print(f'  {i+1}. {feature_type}: center=({center.x:.2f}, {center.y:.2f}), {size_info}, conf={confidence:.3f}, central={is_central}')
        
        if len(feature_set.features) > 10:
            print(f'  ... and {len(feature_set.features) - 10} more features')
            
    except Exception as e:
        print(f'ERROR: {e}')
    
    print()

print('=== TARGET VALIDATION ===')
print('c-bp.dxf target: 4 circles + 1 central circular hole')
print('c_tp.dxf target: 19 circles + 1 central circular hole') 
print('box_front target: 3 circles + 1 square hole')
print('box_rear target: 1 square hole')