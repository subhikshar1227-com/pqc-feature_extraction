from feature_extraction.expected.square_hole_detector import SquareHoleDetector
from feature_extraction.dxf.parser import parse_dxf
from feature_extraction.dxf.geometry_normalizer import normalize_geometry
from feature_extraction.dxf.geometry_reconstruction import analyze_relationships
from pathlib import Path

entities = parse_dxf(Path('dxf/cad_box_front.dxf'))
normalized = normalize_geometry(entities)
relationships = analyze_relationships(normalized)
detector = SquareHoleDetector()

# Full detection pipeline
square_holes = detector.detect_square_holes(normalized, relationships)
print(f'Square holes detected: {len(square_holes)}')

# Debug the pipeline step by step
line_entities = [e for e in normalized if e.source_entity.entity_type.value == 'LINE']
rejection_log = {'not_closed': 0, 'insufficient_sides': 0, 'non_rectangular': 0, 'too_small': 0, 'low_confidence': 0}
closed_polygons = detector._find_closed_line_polygons(line_entities, rejection_log)

print(f'Closed polygons: {len(closed_polygons)}')

if closed_polygons:
    polygon = closed_polygons[0]
    
    # Analyze polygon for hole
    hole_candidate = detector._analyze_polygon_for_hole(polygon, normalized, relationships)
    
    if hole_candidate:
        print(f'Hole candidate created: {hole_candidate.feature_id}')
        print(f'  Confidence: {hole_candidate.confidence:.3f}')
        print(f'  Min confidence required: {detector.min_confidence}')
        
        # Check size constraints
        meets_size = detector._meets_size_constraints(hole_candidate)
        print(f'  Meets size constraints: {meets_size}')
        
        width = hole_candidate.geometric_properties.get('width', 0)
        height = hole_candidate.geometric_properties.get('height', 0)
        min_dimension = min(width, height) if width and height else 0
        print(f'  Width: {width}, Height: {height}, Min dimension: {min_dimension}')
        print(f'  Min size required: {detector.min_size}')
        
    else:
        print('No hole candidate created - polygon analysis failed')
else:
    print('No closed polygons found')