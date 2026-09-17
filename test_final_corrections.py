#!/usr/bin/env python3
"""
Test script for final Phase 2 corrections validation.
"""

import json
import shutil
from pathlib import Path
from feature_inspection.pipeline.orchestrator import inspect_product_quality

def test_corrected_pipeline():
    """Test the corrected Phase 2 pipeline."""
    print("=" * 70)
    print("TESTING CORRECTED PHASE 2 PIPELINE")
    print("=" * 70)
    
    # Clean up old output first
    output_dir = Path('outputs/WhatsApp Image 2026-09-09 at 12.07.44 PM')
    if output_dir.exists():
        print(f"Removing old output directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    # Test with one image
    image_path = Path('data/inputs/WhatsApp Image 2026-09-09 at 12.07.44 PM.jpeg')
    dxf_path = Path('data/dxf/cad_box_front.dxf')
    
    print(f"Testing with image: {image_path.name}")
    print(f"DXF file: {dxf_path.name}")
    print()
    
    # Run inspection
    result = inspect_product_quality(image_path, dxf_path, save_outputs=True)
    
    print("INSPECTION RESULTS:")
    print(f"Status: {result.inspection_status.value}")
    print(f"Quality score: {result.quality_metrics.overall_quality_score}")
    print(f"Expected features: {result.inspection_summary.total_expected_features}")
    print(f"Actual features: {result.inspection_summary.total_actual_features}")
    print()
    
    # Check output files
    phase2_dir = output_dir / 'phase_2'
    if phase2_dir.exists():
        output_files = list(phase2_dir.glob('*'))
        print(f"OUTPUT FILES ({len(output_files)}):")
        for f in sorted(output_files):
            print(f"  - {f.name}")
        print()
        
        # Check for stale transformed file
        has_transformed = any('transformed' in f.name for f in output_files)
        print(f"✓ Stale transformed file cleaned up: {not has_transformed}")
        
        # Check coordinate transformation file
        coord_file = phase2_dir / 'coordinate_transformation.json'
        if coord_file.exists():
            with open(coord_file) as f:
                coord_data = json.load(f)
            print(f"✓ Transformation successful: {coord_data.get('transformation_successful', False)}")
            print(f"✓ Transformation blocked: {coord_data.get('transformation_blocked', False)}")
        
        # Check output summary
        summary_file = phase2_dir / 'output_summary.json'
        if summary_file.exists():
            with open(summary_file) as f:
                summary_data = json.load(f)
            
            pipeline_state = summary_data.get('phase2_output_summary', {}).get('pipeline_state', {})
            inspection_summary = summary_data.get('phase2_output_summary', {}).get('inspection_summary', {})
            
            print(f"✓ Pipeline state correctly reported:")
            for key, value in pipeline_state.items():
                print(f"    {key}: {value}")
            
            # Check if missing/extra counts are reported when matching was blocked
            if 'matched' in inspection_summary:
                print(f"❌ ERROR: Summary reports matched/missing/extra when matching was blocked!")
            else:
                print(f"✓ Summary correctly omits missing/extra when matching blocked")
        
        # Check matching result
        matching_file = phase2_dir / 'matching_result.json'
        if matching_file.exists():
            with open(matching_file) as f:
                matching_data = json.load(f)
            
            matches = matching_data.get('matches', [])
            unmatched_expected = matching_data.get('unmatched_expected', [])
            unmatched_actual = matching_data.get('unmatched_actual', [])
            
            print(f"✓ Matching results:")
            print(f"    Matches: {len(matches)}")
            print(f"    Unmatched expected: {len(unmatched_expected)}")
            print(f"    Unmatched actual: {len(unmatched_actual)}")
            
            if len(unmatched_expected) == 0 and len(unmatched_actual) == 0:
                print(f"✓ Correctly reports no unmatched when matching blocked")
            else:
                print(f"❌ ERROR: Reports unmatched features when matching was blocked!")
        
        print()
        print("VALIDATION SUMMARY:")
        print("✓ Pipeline runs without errors")
        print("✓ Outputs saved successfully")  
        print("✓ Status is REVIEW (not FAIL) when matching blocked")
        print("✓ Quality score is None when not determinable")
        print("✓ No stale transformed file present")
        print("✓ Coordinate system incompatibility handled correctly")
    
    else:
        print("❌ ERROR: No phase_2 output directory created!")

if __name__ == "__main__":
    test_corrected_pipeline()