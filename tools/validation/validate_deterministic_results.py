#!/usr/bin/env python3
"""
Validate that feature extraction produces deterministic results.
"""

import hashlib
import json
from pathlib import Path

import sys
sys.path.append('../../')

from feature_extraction.expected_feature_extractor import extract_expected_features


def serialize_features(feature_set):
    """Convert feature set to deterministic serializable format."""
    features_data = []
    for feature in feature_set.features:
        feature_dict = {
            'type': feature.feature_type.value,
            'center': [round(feature.center.x, 6), round(feature.center.y, 6)],
            'confidence': round(feature.confidence, 6),
            'source_entities': sorted(feature.source_entity_ids),
            'source_type': feature.source_type
        }
        if feature.radius is not None:
            feature_dict['radius'] = round(feature.radius, 6)
        features_data.append(feature_dict)
    
    # Sort features for consistent ordering
    features_data.sort(key=lambda x: (x['center'], x['type']))
    
    return {
        'feature_count': len(features_data),
        'features': features_data,
        'circle_count': feature_set.circle_count,
        'hole_count': feature_set.through_hole_count
    }


def test_deterministic_extraction(dxf_path, runs=3):
    """Test that multiple extractions produce identical results."""
    print(f"Testing deterministic extraction for {Path(dxf_path).name}")
    
    results = []
    hashes = []
    
    for run in range(runs):
        feature_set = extract_expected_features(Path(dxf_path))
        serialized = serialize_features(feature_set)
        json_str = json.dumps(serialized, sort_keys=True)
        hash_value = hashlib.sha256(json_str.encode()).hexdigest()
        
        results.append(serialized)
        hashes.append(hash_value)
        
        print(f"  Run {run + 1}: {serialized['feature_count']} features, hash={hash_value[:16]}...")
    
    # Check if all hashes are identical
    all_identical = len(set(hashes)) == 1
    
    if all_identical:
        print(f"  Result: PASS - All {runs} runs produced identical results")
    else:
        print(f"  Result: FAIL - Runs produced different results")
        for i, (r1, r2) in enumerate(zip(results[:-1], results[1:])):
            if r1 != r2:
                print(f"    Difference between run {i+1} and {i+2}")
    
    return all_identical


def main():
    """Test deterministic results for all DXF files."""
    print("=" * 60)
    print("DETERMINISTIC RESULTS VALIDATION")
    print("=" * 60)
    
    dxf_files = [
        "dxf/c-bp.dxf",
        "dxf/c_tp.dxf", 
        "dxf/cad_box_front.dxf",
        "dxf/cad_box_rear (1).dxf"
    ]
    
    all_passed = True
    
    for dxf_file in dxf_files:
        try:
            result = test_deterministic_extraction(dxf_file)
            if not result:
                all_passed = False
        except Exception as e:
            print(f"  Result: FAIL - Exception: {e}")
            all_passed = False
        print()
    
    print("=" * 60)
    print(f"Overall Result: {'ALL DETERMINISTIC' if all_passed else 'NON-DETERMINISTIC DETECTED'}")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)