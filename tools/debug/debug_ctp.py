import sys
sys.path.append('../../')
from feature_extraction.expected_feature_extractor import extract_expected_features
from pathlib import Path

print('=== C_TP.DXF DETAILED DEBUG ===')
feature_set = extract_expected_features(Path('../../data/dxf/c_tp.dxf'))
print(f'Total features: {feature_set.total_feature_count}')
print(f'Circles: {feature_set.circle_count}, Through holes: {feature_set.through_hole_count}')

# Group features by location to find potential central holes
locations = {}
for i, feature in enumerate(feature_set.features):
    center = feature.center
    location_key = f"{center.x:.0f}_{center.y:.0f}"
    
    if location_key not in locations:
        locations[location_key] = []
    locations[location_key].append((i+1, feature))

print(f'\nFeatures grouped by location:')
for location_key, features_at_location in locations.items():
    if len(features_at_location) > 1:
        print(f'Location {location_key}: {len(features_at_location)} features')
        for idx, feature in features_at_location:
            feature_type = feature.feature_type.value
            radius = getattr(feature, 'radius', 'N/A')
            print(f'  {idx}. {feature_type}: radius={radius}')
    else:
        idx, feature = features_at_location[0]
        feature_type = feature.feature_type.value
        center = feature.center  
        radius = getattr(feature, 'radius', 'N/A')
        is_central = feature.detection_evidence.get('evidence_details', {}).get('central_position', {}).get('is_central', False)
        print(f'{idx}. {feature_type}: center=({center.x:.2f}, {center.y:.2f}), radius={radius}, central={is_central}')

print('\n=== EXPECTED TARGET ===')
print('c_tp.dxf target: 19 circles + 1 central circular hole')
print('Current result: 20 circles + 0 holes')
print('Issue: Need to identify 1 circle as central hole')