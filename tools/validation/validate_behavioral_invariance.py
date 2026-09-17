#!/usr/bin/env python3
"""
Behavioral Invariance Validation for Feature Extraction

Tests that feature detection remains invariant under:
1. Translation
2. Uniform scaling  
3. Rotation (where applicable)
4. Entity ordering
5. Filename independence

Uses synthetic geometry - NOT the four product DXFs.
"""

import os
import tempfile
import math
from pathlib import Path

import ezdxf
from ezdxf import units

import sys
sys.path.append('../../')

from feature_extraction.expected_feature_extractor import extract_expected_features


def create_synthetic_dxf(filename, circles, arcs=None, lines=None, transform=None):
    """Create a synthetic DXF with known geometry."""
    doc = ezdxf.new('R2010', setup=True)
    doc.units = units.MM
    msp = doc.modelspace()
    
    # Add circles
    for center_x, center_y, radius in circles:
        if transform:
            center_x, center_y = transform['translate'](center_x, center_y)
            radius *= transform.get('scale', 1.0)
        msp.add_circle((center_x, center_y), radius)
    
    # Add arcs if specified
    if arcs:
        for center_x, center_y, radius, start_angle, end_angle in arcs:
            if transform:
                center_x, center_y = transform['translate'](center_x, center_y)
                radius *= transform.get('scale', 1.0)
                start_angle += transform.get('rotate', 0)
                end_angle += transform.get('rotate', 0)
            msp.add_arc((center_x, center_y), radius, start_angle, end_angle)
    
    # Add lines if specified  
    if lines:
        for start_x, start_y, end_x, end_y in lines:
            if transform:
                start_x, start_y = transform['translate'](start_x, start_y)
                end_x, end_y = transform['translate'](end_x, end_y)
            msp.add_line((start_x, start_y), (end_x, end_y))
    
    doc.saveas(filename)


def extract_features_from_synthetic(dxf_path):
    """Extract features from synthetic DXF file."""
    feature_set = extract_expected_features(Path(dxf_path))
    return feature_set.features


def compare_feature_sets(features1, features2, tolerance=0.01):
    """Compare two feature sets for geometric equivalence."""
    if len(features1) != len(features2):
        return False, f"Different feature counts: {len(features1)} vs {len(features2)}"
    
    # Sort features by center coordinates for comparison
    def sort_key(f):
        return (round(f.center.x, 2), round(f.center.y, 2), f.feature_type.value)
    
    sorted_f1 = sorted(features1, key=sort_key)
    sorted_f2 = sorted(features2, key=sort_key)
    
    for i, (f1, f2) in enumerate(zip(sorted_f1, sorted_f2)):
        if f1.feature_type != f2.feature_type:
            return False, f"Feature {i}: type mismatch {f1.feature_type.value} vs {f2.feature_type.value}"
        
        # Compare centers (with tolerance for floating point)
        dx = abs(f1.center.x - f2.center.x)
        dy = abs(f1.center.y - f2.center.y)
        if dx > tolerance or dy > tolerance:
            return False, f"Feature {i}: center mismatch ({f1.center.x}, {f1.center.y}) vs ({f2.center.x}, {f2.center.y})"
        
        # Compare radius if present
        if f1.radius is not None and f2.radius is not None:
            if abs(f1.radius - f2.radius) > tolerance:
                return False, f"Feature {i}: radius mismatch {f1.radius} vs {f2.radius}"
        elif (f1.radius is None) != (f2.radius is None):
            return False, f"Feature {i}: radius presence mismatch"
    
    return True, "Features match"
def test_translation_invariance():
    """Test that translation doesn't affect feature detection."""
    print("Testing translation invariance...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Base synthetic geometry: 3 circles at known positions
        base_circles = [(10.0, 10.0, 5.0), (30.0, 10.0, 3.0), (20.0, 25.0, 4.0)]
        
        # Original file
        original_file = Path(tmpdir) / "original.dxf"
        create_synthetic_dxf(original_file, base_circles)
        
        # Translated file
        translation = {'translate': lambda x, y: (x + 100, y + 50)}
        translated_file = Path(tmpdir) / "translated.dxf"
        create_synthetic_dxf(translated_file, base_circles, transform=translation)
        
        # Extract features
        original_features = extract_features_from_synthetic(original_file)
        translated_features = extract_features_from_synthetic(translated_file)
        
        # Adjust translated features back to original coordinate system
        for feature in translated_features:
            # Create new Point2D with adjusted coordinates
            from feature_extraction.dxf.entity_models import Point2D
            feature.center = Point2D(feature.center.x - 100, feature.center.y - 50)
        
        match, msg = compare_feature_sets(original_features, translated_features)
        
        print(f"  Original features: {len(original_features)}")
        print(f"  Translated features: {len(translated_features)}")
        print(f"  Result: {'PASS' if match else 'FAIL'} - {msg}")
        
        return match


def test_scaling_invariance():
    """Test that uniform scaling preserves feature relationships."""
    print("Testing scaling invariance...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Base synthetic geometry
        base_circles = [(10.0, 10.0, 5.0), (30.0, 10.0, 3.0)]
        
        # Original file
        original_file = Path(tmpdir) / "original.dxf"
        create_synthetic_dxf(original_file, base_circles)
        
        # Scaled file (2x scaling)
        scaling = {
            'translate': lambda x, y: (x * 2.0, y * 2.0),
            'scale': 2.0
        }
        scaled_file = Path(tmpdir) / "scaled.dxf"
        create_synthetic_dxf(scaled_file, base_circles, transform=scaling)
        
        # Extract features
        original_features = extract_features_from_synthetic(original_file)
        scaled_features = extract_features_from_synthetic(scaled_file)
        
        # Adjust scaled features back to original coordinate system
        for feature in scaled_features:
            from feature_extraction.dxf.entity_models import Point2D
            feature.center = Point2D(feature.center.x / 2.0, feature.center.y / 2.0)
            if feature.radius is not None:
                feature.radius /= 2.0
        
        match, msg = compare_feature_sets(original_features, scaled_features)
        
        print(f"  Original features: {len(original_features)}")
        print(f"  Scaled features: {len(scaled_features)}")
        print(f"  Result: {'PASS' if match else 'FAIL'} - {msg}")
        
        return match


def test_entity_ordering_invariance():
    """Test that entity order doesn't affect detection."""
    print("Testing entity ordering invariance...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        base_circles = [(10.0, 10.0, 5.0), (30.0, 10.0, 3.0), (20.0, 25.0, 4.0)]
        
        # Create two files with same geometry but different entity order
        file1 = Path(tmpdir) / "ordered1.dxf"
        file2 = Path(tmpdir) / "ordered2.dxf" 
        
        # File 1: normal order
        create_synthetic_dxf(file1, base_circles)
        
        # File 2: reversed order
        create_synthetic_dxf(file2, list(reversed(base_circles)))
        
        features1 = extract_features_from_synthetic(file1)
        features2 = extract_features_from_synthetic(file2)
        
        match, msg = compare_feature_sets(features1, features2)
        
        print(f"  File 1 features: {len(features1)}")
        print(f"  File 2 features: {len(features2)}")
        print(f"  Result: {'PASS' if match else 'FAIL'} - {msg}")
        
        return match


def test_filename_independence():
    """Test that filename doesn't affect feature detection."""
    print("Testing filename independence...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        base_circles = [(10.0, 10.0, 5.0), (30.0, 10.0, 3.0)]
        
        # Same geometry, different filenames
        file1 = Path(tmpdir) / "test_file_A.dxf"
        file2 = Path(tmpdir) / "completely_different_name_B.dxf"
        
        create_synthetic_dxf(file1, base_circles)
        create_synthetic_dxf(file2, base_circles)
        
        features1 = extract_features_from_synthetic(file1)
        features2 = extract_features_from_synthetic(file2)
        
        match, msg = compare_feature_sets(features1, features2)
        
        print(f"  File 1 features: {len(features1)}")
        print(f"  File 2 features: {len(features2)}")
        print(f"  Result: {'PASS' if match else 'FAIL'} - {msg}")
        
        return match


def test_rotation_invariance():
    """Test rotation invariance for circular features."""
    print("Testing rotation invariance (circles only)...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        base_circles = [(0.0, 0.0, 5.0), (10.0, 0.0, 3.0)]  # Centered for easier rotation
        
        # Original file
        original_file = Path(tmpdir) / "original.dxf"
        create_synthetic_dxf(original_file, base_circles)
        
        # Rotated file (45 degrees)
        angle = math.pi / 4  # 45 degrees in radians
        rotation = {
            'translate': lambda x, y: (
                x * math.cos(angle) - y * math.sin(angle),
                x * math.sin(angle) + y * math.cos(angle)
            ),
            'rotate': math.degrees(angle)
        }
        rotated_file = Path(tmpdir) / "rotated.dxf"
        create_synthetic_dxf(rotated_file, base_circles, transform=rotation)
        
        # Extract features
        original_features = extract_features_from_synthetic(original_file)
        rotated_features = extract_features_from_synthetic(rotated_file)
        
        # For circles, rotation shouldn't change the count or radii
        original_count = len(original_features)
        rotated_count = len(rotated_features)
        
        # Compare radii (rotation shouldn't affect radius)
        original_radii = sorted([f.radius for f in original_features if f.radius is not None])
        rotated_radii = sorted([f.radius for f in rotated_features if f.radius is not None])
        
        count_match = original_count == rotated_count
        radii_match = len(original_radii) == len(rotated_radii) and all(
            abs(r1 - r2) < 0.01 for r1, r2 in zip(original_radii, rotated_radii)
        )
        
        success = count_match and radii_match
        
        print(f"  Original features: {original_count}")
        print(f"  Rotated features: {rotated_count}")
        print(f"  Original radii: {original_radii}")
        print(f"  Rotated radii: {rotated_radii}")
        print(f"  Result: {'PASS' if success else 'FAIL'}")
        
        return success


def main():
    """Run all behavioral invariance tests."""
    print("=" * 60)
    print("BEHAVIORAL INVARIANCE VALIDATION")
    print("=" * 60)
    print("Using synthetic geometry (NOT product DXFs)")
    print()
    
    tests = [
        test_translation_invariance,
        test_scaling_invariance,
        test_entity_ordering_invariance,
        test_filename_independence,
        test_rotation_invariance,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"  Result: FAIL - Exception: {e}")
            results.append(False)
            print()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    test_names = [
        "Translation Invariance",
        "Scaling Invariance", 
        "Entity Ordering Invariance",
        "Filename Independence",
        "Rotation Invariance"
    ]
    
    all_passed = True
    for name, result in zip(test_names, results):
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
        if not result:
            all_passed = False
    
    print()
    print(f"Overall Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)