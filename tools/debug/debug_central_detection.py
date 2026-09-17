#!/usr/bin/env python3
"""
Debug Central Hole Detection

Analyzes why circles at (148.5, 105.0) in c-bp and c_tp are not being 
detected as central holes.
"""

from feature_extraction.dxf import parse_dxf, normalize_geometry, analyze_relationships
from feature_extraction.expected import ThroughHoleDetector
from pathlib import Path

def debug_central_detection(dxf_file: str):
    """Debug central hole detection for a specific DXF file."""
    
    print(f"\n{'='*60}")
    print(f"DEBUGGING CENTRAL DETECTION: {dxf_file}")
    print(f"{'='*60}")
    
    # Load and process DXF
    entities = parse_dxf(Path(f'dxf/{dxf_file}'))
    normalized = normalize_geometry(entities)
    relationships = analyze_relationships(normalized)
    
    print(f"Total entities: {len(normalized)}")
    
    # Find circles at geometric center (148.5, 105.0)
    target_center = (148.5, 105.0)
    central_circles = []
    
    for entity in normalized:
        if (entity.source_entity.entity_type.name == 'CIRCLE' and 
            entity.source_entity.center and 
            abs(entity.source_entity.center.x - target_center[0]) < 0.1 and
            abs(entity.source_entity.center.y - target_center[1]) < 0.1):
            central_circles.append(entity)
    
    print(f"\nCircles at geometric center {target_center}:")
    for i, circle in enumerate(central_circles):
        print(f"  {i+1}. {circle.source_entity.entity_id}: radius={circle.source_entity.radius:.2f}")
    
    if not central_circles:
        print("  No circles found at geometric center!")
        return
    
    # Test central detection on each circle
    detector = ThroughHoleDetector()
    
    for circle in central_circles:
        print(f"\n--- ANALYZING {circle.source_entity.entity_id} (radius={circle.source_entity.radius:.2f}) ---")
        
        # Get geometric context
        geometric_context = detector._analyze_geometric_context(circle, normalized)
        
        print(f"Geometric context:")
        print(f"  Main body center: {geometric_context.get('main_body_center')}")
        print(f"  Main body radius: {geometric_context.get('main_body_radius')}")
        print(f"  Large circles: {len(geometric_context.get('large_circles', []))}")
        print(f"  Feature circles: {len(geometric_context.get('feature_circles', []))}")
        print(f"  Total circular entities: {geometric_context.get('circular_entities')}")
        
        # Test hole evidence analysis
        try:
            evidence_analysis = detector._analyze_comprehensive_hole_evidence(
                circle, normalized, relationships
            )
            
            print(f"\nHole evidence analysis:")
            print(f"  Total confidence: {evidence_analysis['total_confidence']:.3f}")
            print(f"  Threshold: {detector.min_confidence}")
            print(f"  Would be detected as hole: {evidence_analysis['total_confidence'] >= detector.min_confidence}")
            
            # Break down evidence
            details = evidence_analysis.get('evidence_details', {})
            print(f"  Evidence breakdown:")
            print(f"    Closure: {evidence_analysis['geometric_closure']:.3f}")
            print(f"    Nesting: {evidence_analysis['concentric_nesting']:.3f}")
            print(f"    Pattern: {evidence_analysis['pattern_regularity']:.3f}")
            print(f"    Size: {evidence_analysis['size_appropriateness']:.3f}")
            print(f"    Layer: {evidence_analysis['layer_context']:.3f}")
            
            # Central position details
            if 'central_position' in details:
                central = details['central_position']
                print(f"    Central position:")
                print(f"      Is central: {central.get('is_central', False)}")
                print(f"      Distance: {central.get('distance_from_center', 'N/A')}")
                print(f"      Evidence score: {details.get('central_position', {}).get('evidence_score', 'N/A')}")
                if 'geometric_centroid' in central:
                    print(f"      Calculated centroid: {central['geometric_centroid']}")
        
        except Exception as e:
            print(f"ERROR analyzing hole evidence: {e}")


if __name__ == "__main__":
    debug_central_detection('c-bp.dxf')
    debug_central_detection('c_tp.dxf')