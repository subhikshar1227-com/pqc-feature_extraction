from feature_extraction.expected.square_hole_detector import SquareHoleDetector
from feature_extraction.dxf.parser import parse_dxf
from feature_extraction.dxf.geometry_normalizer import normalize_geometry
from pathlib import Path

entities = parse_dxf(Path('dxf/cad_box_front.dxf'))
normalized = normalize_geometry(entities)
detector = SquareHoleDetector()
line_entities = [e for e in normalized if e.source_entity.entity_type.value == 'LINE']

rejection_log = {'not_closed': 0, 'insufficient_sides': 0}
closed_polygons = detector._find_closed_line_polygons(line_entities, rejection_log)

if closed_polygons:
    polygon = closed_polygons[0]
    vertices = detector._extract_ordered_vertices(polygon)
    print('Vertices:')
    for i, v in enumerate(vertices):
        print(f'{i+1}: ({v.x:.2f}, {v.y:.2f})')
    
    geom_analysis = detector._analyze_polygon_geometry(vertices)
    print(f'Rectangular: {geom_analysis.get("is_rectangular")}')
    print(f'Square: {geom_analysis.get("is_square")}') 
    print(f'Width: {geom_analysis.get("width", 0):.2f}')
    print(f'Height: {geom_analysis.get("height", 0):.2f}')
    if 'side_lengths' in geom_analysis:
        print(f'Side lengths: {[round(l, 2) for l in geom_analysis["side_lengths"]]}')
    if 'angles' in geom_analysis:
        print(f'Angles: {[round(a, 1) for a in geom_analysis["angles"]]}')