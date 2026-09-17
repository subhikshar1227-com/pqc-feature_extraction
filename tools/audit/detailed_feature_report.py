#!/usr/bin/env python3
"""
Detailed Feature Report Generator

Shows exactly which DXF geometry produced each final feature.
NO hardcoded targets or expectations - pure geometry-to-feature mapping.
"""

from pathlib import Path
from feature_extraction import extract_expected_features


def generate_detailed_report():
    """Generate detailed geometry-to-feature mapping for all DXF files."""
    
    print("=" * 80)
    print("DETAILED GEOMETRY-TO-FEATURE MAPPING REPORT")
    print("=" * 80)
    print("Shows exactly which DXF geometry produced each detected feature")
    print("NO hardcoded expectations - pure geometric analysis results")
    print()
    
    dxf_dir = Path("dxf")
    dxf_files = sorted(dxf_dir.glob("*.dxf"))
    
    for dxf_file in dxf_files:
        print(f"\n{'='*60}")
        print(f"DXF FILE: {dxf_file.name}")
        print(f"{'='*60}")
        
        try:
            # Extract features with full traceability
            feature_set = extract_expected_features(dxf_file)
            
            print(f"SUMMARY: {feature_set.total_feature_count} total features")
            print(f"  Circles: {feature_set.circle_count}")
            print(f"  Through holes: {feature_set.through_hole_count}")
            print(f"  Average confidence: {feature_set.average_confidence:.3f}")
            print()
            
            if not feature_set.features:
                print("  No features detected in this file")
                continue
            
            print("DETAILED FEATURE MAPPING:")
            print("-" * 40)
            
            for i, feature in enumerate(feature_set.features, 1):
                print(f"\n[{i}] FEATURE: {feature.feature_id}")
                print(f"    Type: {feature.feature_type.value}")
                print(f"    Confidence: {feature.confidence:.3f}")
                print(f"    Center: ({feature.center.x:.3f}, {feature.center.y:.3f})")
                
                if feature.radius is not None:
                    print(f"    Radius: {feature.radius:.3f}")
                
                print(f"    Source Type: {feature.source_type}")
                print(f"    Source Entity IDs: {feature.source_entity_ids}")
                
                # Show geometric properties
                if feature.geometric_properties:
                    print("    Geometric Properties:")
                    for key, value in feature.geometric_properties.items():
                        print(f"      {key}: {value}")
                
                # Show detection evidence
                if feature.detection_evidence:
                    print("    Detection Evidence:")
                    evidence = feature.detection_evidence
                    
                    # Show key evidence scores
                    if "total_confidence" in evidence:
                        print(f"      Total confidence: {evidence['total_confidence']:.3f}")
                    
                    if "geometric_analysis" in evidence:
                        geom = evidence["geometric_analysis"]
                        print(f"      Geometric analysis:")
                        for key in ["is_rectangular", "is_square", "width", "height", "area"]:
                            if key in geom:
                                print(f"        {key}: {geom[key]}")
                    
                    if "validation_method" in evidence:
                        print(f"      Validation method: {evidence['validation_method']}")
                    
                    # For through holes, show hole evidence breakdown
                    if "evidence_details" in evidence:
                        details = evidence["evidence_details"]
                        if any(key in details for key in ["closure", "nesting", "pattern", "size", "central_position"]):
                            print("      Evidence breakdown:")
                            
                            if "closure" in details:
                                closure = details["closure"]
                                print(f"        Closure: {closure.get('entity_type', 'unknown')} "
                                      f"(coverage: {closure.get('coverage', 0):.2f})")
                            
                            if "central_position" in details:
                                central = details["central_position"]
                                is_central = central.get("is_central", False)
                                distance = central.get("distance_from_center", "unknown")
                                print(f"        Central position: {is_central} (distance: {distance})")
                            
                            if "pattern" in details:
                                pattern = details["pattern"]
                                similar_count = pattern.get("similar_count", 0)
                                print(f"        Pattern: {similar_count} similar entities")
            
            # Show reference geometry summary
            if hasattr(feature_set, 'reference_geometry') and feature_set.reference_geometry:
                print(f"\nREFERENCE GEOMETRY SUMMARY:")
                print("-" * 30)
                
                entity_types = {}
                for entity in feature_set.reference_geometry:
                    entity_type = entity.source_entity.entity_type.value
                    entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                
                print(f"  Total entities: {len(feature_set.reference_geometry)}")
                for entity_type, count in sorted(entity_types.items()):
                    print(f"    {entity_type}: {count}")
            
            # Show reconstructed geometry summary
            if (hasattr(feature_set, 'reconstructed_geometry') and 
                feature_set.reconstructed_geometry and 
                hasattr(feature_set.reconstructed_geometry, 'reconstructed_circles')):
                
                recon_circles = feature_set.reconstructed_geometry.reconstructed_circles
                if recon_circles:
                    print(f"\nRECONSTRUCTED GEOMETRY:")
                    print("-" * 25)
                    print(f"  Reconstructed circles: {len(recon_circles)}")
                    for circle in recon_circles:
                        print(f"    {circle.circle_id}: center=({circle.center.x:.1f}, {circle.center.y:.1f}), "
                              f"radius={circle.radius:.1f}, coverage={circle.coverage_fraction:.1%}")
                        
        except Exception as e:
            print(f"ERROR processing {dxf_file.name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    generate_detailed_report()