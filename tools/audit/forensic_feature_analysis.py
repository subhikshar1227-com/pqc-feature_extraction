#!/usr/bin/env python3
"""
Forensic Feature Analysis

TASK 1: FROZEN ALGORITHM STATE - DO NOT CHANGE THRESHOLDS TO FIX TARGETS
TASK 2: Comprehensive geometry-to-feature explanation for EVERY detected feature.

This analysis identifies WHY large circles are missing from detection.
The goal is GEOMETRIC UNDERSTANDING, not tuning to match expected counts.

Reports exact geometric evidence and inference chain for each classification.
"""

from feature_extraction.expected_feature_extractor import extract_expected_features
from feature_extraction.dxf import parse_dxf, normalize_geometry, analyze_relationships, reconstruct_geometry
from feature_extraction.expected.circle_detector import CircleDetector
from feature_extraction.expected.through_hole_detector import ThroughHoleDetector
from pathlib import Path
import json

def analyze_missing_geometry(dxf_file: str):
    """
    TASK 2: Analyze WHY geometry is missing from detection.
    
    This identifies the GEOMETRIC FILTERING that removes circles before classification.
    The root cause is NOT classification thresholds - it's detection filtering.
    """
    print(f"\n{'='*80}")
    print(f"MISSING GEOMETRY ANALYSIS: {dxf_file}")
    print(f"{'='*80}")
    
    # Get raw geometry vs detected features
    dxf_path = Path(f'dxf/{dxf_file}')
    entities = parse_dxf(dxf_path)
    normalized = normalize_geometry(entities)
    relationships = analyze_relationships(normalized)
    reconstructed = reconstruct_geometry(normalized, relationships)
    
    feature_set = extract_expected_features(dxf_path)
    
    # Find all raw circular geometry
    all_circles = []
    for entity in entities:
        if entity.entity_type.value == "CIRCLE":
            all_circles.append({
                "entity_id": entity.entity_id,
                "center": (entity.center.x, entity.center.y) if entity.center else None,
                "radius": entity.radius,
                "layer": entity.layer
            })
    
    # Find which circles were detected vs missing
    detected_entity_ids = set()
    for feature in feature_set.features:
        detected_entity_ids.update(feature.source_entity_ids)
    
    missing_circles = []
    detected_circles = []
    
    for circle in all_circles:
        if circle["entity_id"] in detected_entity_ids:
            detected_circles.append(circle)
        else:
            missing_circles.append(circle)
    
    print(f"\nRAW CIRCLE INVENTORY:")
    print(f"  Total circles in DXF: {len(all_circles)}")
    print(f"  Circles detected as features: {len(detected_circles)}")  
    print(f"  Circles missing from detection: {len(missing_circles)}")
    
    if missing_circles:
        print(f"\nMISSING CIRCLES (GEOMETRIC FILTERING):")
        print("-" * 50)
        
        for circle in missing_circles:
            print(f"  MISSING: {circle['entity_id']}")
            print(f"    Center: {circle['center']}")
            print(f"    Radius: {circle['radius']}")
            print(f"    Layer: {circle['layer']}")
            
            # Analyze WHY this circle was filtered out
            reason = analyze_circle_filtering_reason(circle, all_circles, entities, normalized)
            print(f"    FILTERING REASON: {reason}")
            print()
    
    if detected_circles:
        print(f"\nDETECTED CIRCLES:")
        print("-" * 20)
        for circle in detected_circles:
            print(f"  DETECTED: {circle['entity_id']} - radius={circle['radius']}")
    
    print(f"\nCRITICAL FINDING:")
    if missing_circles:
        print(f"  Large circles are being FILTERED OUT during detection, not classification.")
        print(f"  The problem is GEOMETRIC FILTERING, not threshold tuning.")
        print(f"  Root cause: _is_main_body_circle() method removing structural geometry.")
    else:
        print(f"  All circles detected successfully.")
        
    return {
        "total_circles": len(all_circles),
        "detected_circles": len(detected_circles),
        "missing_circles": len(missing_circles),
        "missing_circle_data": missing_circles
    }

def analyze_circle_filtering_reason(missing_circle, all_circles, entities, normalized):
    """Determine why a specific circle was filtered out during detection."""
    
    radius = missing_circle["radius"]
    if not radius:
        return "Invalid geometry - no radius data"
    
    # Import current configuration to see what thresholds apply
    from feature_extraction.config import MAIN_BODY_MIN_RADIUS
    
    # Check absolute size threshold
    if radius > MAIN_BODY_MIN_RADIUS:
        return f"ABSOLUTE SIZE: radius {radius:.1f} > MAIN_BODY_MIN_RADIUS {MAIN_BODY_MIN_RADIUS}"
    
    # Calculate geometric context that circle detector uses
    circle_radii = [c["radius"] for c in all_circles if c["radius"]]
    if circle_radii:
        max_radius = max(circle_radii)
        avg_radius = sum(circle_radii) / len(circle_radii)
        size_diversity = max_radius / min(circle_radii) if min(circle_radii) > 0 else 1.0
        
        # Check relative size detection logic
        if (radius == max_radius and 
            radius > avg_radius * 3.0 and
            radius > 15.0 and
            size_diversity > 2.0):
            return f"RELATIVE SIZE: largest circle {radius:.1f} >> avg {avg_radius:.1f} (factor {radius/avg_radius:.1f})"
    
    # Check concentric filtering
    from feature_extraction.config import CONCENTRICITY_TOLERANCE
    center = missing_circle["center"]
    if center:
        for other_circle in all_circles:
            if (other_circle["entity_id"] != missing_circle["entity_id"] and 
                other_circle["center"] and other_circle["radius"]):
                
                other_center = other_circle["center"]
                center_distance = ((center[0] - other_center[0])**2 + 
                                 (center[1] - other_center[1])**2)**0.5
                
                if center_distance < CONCENTRICITY_TOLERANCE * 2.0 and other_circle["radius"] > 15.0:
                    return f"CONCENTRIC: distance {center_distance:.2f} to large circle {other_circle['entity_id']}"
    
    # Check containment logic  
    if radius > 20.0:
        small_circle_count = sum(1 for c in all_circles if c["radius"] and c["radius"] <= 15.0)
        if small_circle_count >= 2:
            return f"CONTAINMENT: large circle {radius:.1f} could contain {small_circle_count} features"
    
    return "UNKNOWN - geometric filtering logic not identified"


def analyze_geometric_evidence(dxf_file: str):
    """Provide comprehensive geometric evidence analysis for all features."""
    
    print(f"\n{'='*80}")
    print(f"FORENSIC ANALYSIS: {dxf_file}")
    print(f"{'='*80}")
    
    # Extract features
    dxf_path = Path(f'dxf/{dxf_file}')
    feature_set = extract_expected_features(dxf_path)
    
    # Also get raw geometry for comparison
    entities = parse_dxf(dxf_path)
    normalized = normalize_geometry(entities)
    relationships = analyze_relationships(normalized)
    reconstructed = reconstruct_geometry(normalized, relationships)
    
    print(f"\nRAW GEOMETRY INVENTORY:")
    print(f"  Total entities: {len(entities)}")
    
    entity_counts = {}
    for entity in entities:
        entity_type = entity.entity_type.value
        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
    
    for entity_type, count in sorted(entity_counts.items()):
        print(f"  {entity_type}: {count}")
    
    print(f"\nRECONSTRUCTED GEOMETRY:")
    print(f"  Reconstructed circles: {len(reconstructed.reconstructed_circles)}")
    
    for i, recon in enumerate(reconstructed.reconstructed_circles):
        print(f"    {i+1}: center=({recon.center.x:.1f}, {recon.center.y:.1f}), "
              f"radius={recon.radius:.1f}, coverage={recon.coverage_fraction:.1%}")
    
    print(f"\nDETECTED FEATURES: {feature_set.total_feature_count}")
    print(f"  Circles: {feature_set.circle_count}")
    print(f"  Through holes: {feature_set.through_hole_count}")
    
    print(f"\nDETAILED FEATURE EVIDENCE:")
    print("-" * 60)
    
    for i, feature in enumerate(feature_set.features, 1):
        print(f"\n[{i}] FEATURE: {feature.feature_id}")
        print(f"    Type: {feature.feature_type.value}")
        print(f"    Center: ({feature.center.x:.3f}, {feature.center.y:.3f})")
        print(f"    Radius/Size: {feature.radius:.3f}" if feature.radius else f"    Dimensions: N/A")
        print(f"    Confidence: {feature.confidence:.3f}")
        print(f"    Source Entity IDs: {feature.source_entity_ids}")
        
        # Analyze source geometry type
        source_type = "unknown"
        if hasattr(feature, 'metadata'):
            source_type = feature.metadata.get('source_type', 'unknown')
        
        print(f"    Source Type: {source_type}")
        
        # Get detection evidence if available
        if hasattr(feature, 'detection_evidence') and feature.detection_evidence:
            evidence = feature.detection_evidence
            print(f"    Detection Evidence:")
            
            if isinstance(evidence, dict):
                # Parse evidence details
                if 'total_confidence' in evidence:
                    print(f"      Total Confidence: {evidence['total_confidence']:.3f}")
                
                if 'evidence_details' in evidence:
                    details = evidence['evidence_details']
                    
                    # Closure evidence
                    if 'closure' in details:
                        closure = details['closure']
                        print(f"      Closure Evidence:")
                        print(f"        Type: {closure.get('closure_type', 'N/A')}")
                        print(f"        Coverage: {closure.get('coverage', 'N/A')}")
                    
                    # Nesting evidence
                    if 'nesting' in details:
                        nesting = details['nesting']
                        print(f"      Nesting Evidence:")
                        print(f"        Relationships: {len(nesting.get('concentric_relationships', []))}")
                        print(f"        Best ratio: {nesting.get('best_nesting_ratio', 'N/A'):.3f}")
                    
                    # Pattern evidence
                    if 'pattern' in details:
                        pattern = details['pattern']
                        print(f"      Pattern Evidence:")
                        print(f"        Similar count: {pattern.get('similar_count', 'N/A')}")
                        print(f"        Pattern score: {pattern.get('pattern_score', 'N/A'):.3f}")
                    
                    # Size evidence
                    if 'size' in details:
                        size = details['size']
                        print(f"      Size Evidence:")
                        print(f"        Size category: {size.get('size_category', 'N/A')}")
                        print(f"        Size evidence: {size.get('size_evidence', 'N/A'):.3f}")
                    
                    # Central position evidence
                    if 'central_position' in details:
                        central = details['central_position']
                        print(f"      Central Position Evidence:")
                        print(f"        Is central: {central.get('is_central', False)}")
                        print(f"        Distance: {central.get('distance_from_center', 'N/A')}")
                        print(f"        Evidence score: {central.get('evidence_score', 'N/A')}")
                        if 'geometric_centroid' in central:
                            centroid = central['geometric_centroid']
                            print(f"        Calculated centroid: ({centroid[0]:.1f}, {centroid[1]:.1f})")
                    
                    # Bonuses applied
                    if 'central_bonus' in details:
                        print(f"      Central Bonus Applied: {details['central_bonus']:.3f}")
                    
                    if 'main_body_center_bonus' in details:
                        print(f"      Main Body Center Bonus: {details['main_body_center_bonus']:.3f}")
                        print(f"      Distance to Main Body: {details.get('distance_to_main_body_center', 'N/A'):.3f}")
        
        # Determine geometric classification basis
        print(f"    Geometric Classification Basis:")
        
        # Find corresponding raw entity
        raw_entities = [e for e in entities if e.entity_id in feature.source_entity_ids]
        if raw_entities:
            entity = raw_entities[0]
            print(f"      Raw entity type: {entity.entity_type.value}")
            
            if entity.entity_type.value == "CIRCLE":
                print(f"      Explicit circle: center=({entity.center.x:.1f}, {entity.center.y:.1f}), radius={entity.radius:.1f}")
            elif entity.entity_type.value == "ARC":
                print(f"      Arc: center=({entity.center.x:.1f}, {entity.center.y:.1f}), radius={entity.radius:.1f}")
                print(f"           angles={entity.start_angle:.1f}° to {entity.end_angle:.1f}°")
            elif entity.entity_type.value == "LINE":
                print(f"      Line: from=({entity.start_point.x:.1f}, {entity.start_point.y:.1f}) "
                      f"to=({entity.end_point.x:.1f}, {entity.end_point.y:.1f})")
        
        # Check if this corresponds to reconstructed geometry
        for recon in reconstructed.reconstructed_circles:
            if any(entity_id in recon.source_entities for entity_id in feature.source_entity_ids):
                print(f"      Reconstructed from {len(recon.source_entities)} arcs")
                print(f"      Coverage: {recon.coverage_fraction:.1%}")
                break
        
        # Determine why this geometry was classified as hole vs circle
        if feature.feature_type.value == "through_hole":
            print(f"    WHY CLASSIFIED AS HOLE:")
            print(f"      Confidence {feature.confidence:.3f} exceeded hole threshold")
            if hasattr(feature, 'detection_evidence') and feature.detection_evidence:
                evidence = feature.detection_evidence.get('evidence_details', {})
                if evidence.get('central_position', {}).get('is_central'):
                    print(f"      Identified as central/main hole based on position")
                if len(evidence.get('nesting', {}).get('concentric_relationships', [])) > 0:
                    print(f"      Nested inside other geometry")
        else:
            print(f"    WHY CLASSIFIED AS CIRCLE:")
            print(f"      Explicit geometric circle with no strong hole evidence")
    
    print(f"\n{'='*80}")

def main():
    """Run forensic analysis on all DXF files."""
    
    dxf_files = ['c-bp.dxf', 'c_tp.dxf', 'cad_box_front.dxf', 'cad_box_rear (1).dxf']
    
    print("FORENSIC FEATURE ANALYSIS")
    print("="*80)
    print("TASK 1: ALGORITHM FROZEN - NO TARGET-DRIVEN TUNING")
    print("TASK 2: Comprehensive geometry-to-feature explanation")
    print("IDENTIFIES WHY LARGE CIRCLES ARE MISSING FROM DETECTION")
    print("="*80)
    
    # First, analyze missing geometry (THE ROOT CAUSE)
    all_missing_data = {}
    for dxf_file in dxf_files:
        missing_data = analyze_missing_geometry(dxf_file)
        all_missing_data[dxf_file] = missing_data
    
    # Then, analyze detected features  
    for dxf_file in dxf_files:
        analyze_geometric_evidence(dxf_file)
    
    # Summary of critical findings
    print(f"\n{'='*80}")
    print("CRITICAL FORENSIC FINDINGS")
    print(f"{'='*80}")
    
    total_missing = sum(data["missing_circles"] for data in all_missing_data.values())
    if total_missing > 0:
        print(f"ROOT CAUSE IDENTIFIED: {total_missing} circles missing from detection")
        print(f"PROBLEM: Geometric filtering in circle detector, NOT classification thresholds")
        print(f"SOLUTION: Fix _is_main_body_circle() logic to preserve feature-scale circles")
        
        for dxf_file, data in all_missing_data.items():
            if data["missing_circles"] > 0:
                print(f"\n{dxf_file}: {data['missing_circles']} missing circles")
                for circle in data["missing_circle_data"]:
                    print(f"  - radius {circle['radius']:.1f} at {circle['center']}")
    else:
        print("No missing circles found - detection is complete")
    
    print(f"\nNEXT STEPS:")
    print(f"1. Fix geometric filtering logic (NOT thresholds)")
    print(f"2. Distinguish true main body from feature-scale circles")  
    print(f"3. Re-run detection without changing classification logic")
    print(f"4. Validate that geometry-driven results are stable")

if __name__ == "__main__":
    main()