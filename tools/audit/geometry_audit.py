#!/usr/bin/env python3
"""
Geometry Audit Utility

Completely generic geometry analysis of DXF files.
NO product-specific coordinates, counts, or expectations.
Pure geometric analysis to understand what exists in each DXF.
"""

import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

from feature_extraction.dxf import parse_dxf, normalize_geometry, analyze_relationships, reconstruct_geometry
from feature_extraction.dxf.entity_models import EntityType, Point2D


def calculate_bounding_box(entities) -> Dict:
    """Calculate bounding box from all entities."""
    if not entities:
        return {"min_x": 0, "max_x": 0, "min_y": 0, "max_y": 0, "width": 0, "height": 0, "center_x": 0, "center_y": 0}
    
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    
    for entity in entities:
        # Get all relevant points from the entity
        points = []
        
        if entity.center:
            if entity.radius:
                # Circle/arc - include bounding box
                points.extend([
                    (entity.center.x - entity.radius, entity.center.y - entity.radius),
                    (entity.center.x + entity.radius, entity.center.y + entity.radius)
                ])
            else:
                points.append((entity.center.x, entity.center.y))
        
        if entity.start_point:
            points.append((entity.start_point.x, entity.start_point.y))
        if entity.end_point:
            points.append((entity.end_point.x, entity.end_point.y))
            
        # Update bounds
        for x, y in points:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
    
    width = max_x - min_x
    height = max_y - min_y
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    return {
        "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y,
        "width": width, "height": height, "center_x": center_x, "center_y": center_y
    }


def analyze_circles(entities) -> Dict:
    """Analyze all CIRCLE entities."""
    circles = [e for e in entities if e.entity_type == EntityType.CIRCLE]
    
    analysis = {
        "count": len(circles),
        "circles": [],
        "radii": [],
        "centers": []
    }
    
    for circle in circles:
        if circle.center and circle.radius:
            circle_info = {
                "entity_id": circle.entity_id,
                "center": (round(circle.center.x, 3), round(circle.center.y, 3)),
                "radius": round(circle.radius, 3),
                "layer": circle.layer,
                "handle": circle.handle
            }
            analysis["circles"].append(circle_info)
            analysis["radii"].append(circle.radius)
            analysis["centers"].append((circle.center.x, circle.center.y))
    
    if analysis["radii"]:
        analysis["radius_stats"] = {
            "min": min(analysis["radii"]),
            "max": max(analysis["radii"]),
            "avg": sum(analysis["radii"]) / len(analysis["radii"])
        }
    
    return analysis


def analyze_arcs(entities) -> Dict:
    """Analyze all ARC entities."""
    arcs = [e for e in entities if e.entity_type == EntityType.ARC]
    
    analysis = {
        "count": len(arcs),
        "arcs": [],
        "by_center": defaultdict(list),
        "by_radius": defaultdict(list)
    }
    
    for arc in arcs:
        if arc.center and arc.radius and arc.start_angle is not None and arc.end_angle is not None:
            # Calculate angular span
            span = arc.end_angle - arc.start_angle
            if span < 0:
                span += 360
                
            arc_info = {
                "entity_id": arc.entity_id,
                "center": (round(arc.center.x, 3), round(arc.center.y, 3)),
                "radius": round(arc.radius, 3),
                "start_angle": round(arc.start_angle, 1),
                "end_angle": round(arc.end_angle, 1),
                "span": round(span, 1),
                "layer": arc.layer,
                "handle": arc.handle
            }
            analysis["arcs"].append(arc_info)
            
            # Group by center for reconstruction analysis
            center_key = (round(arc.center.x, 1), round(arc.center.y, 1))
            analysis["by_center"][center_key].append(arc_info)
            
            # Group by radius for similarity analysis
            radius_key = round(arc.radius, 1)
            analysis["by_radius"][radius_key].append(arc_info)
    
    return analysis


def analyze_lines(entities) -> Dict:
    """Analyze all LINE entities."""
    lines = [e for e in entities if e.entity_type == EntityType.LINE]
    
    analysis = {
        "count": len(lines),
        "lines": []
    }
    
    for line in lines:
        if line.start_point and line.end_point:
            dx = line.end_point.x - line.start_point.x
            dy = line.end_point.y - line.start_point.y
            length = math.sqrt(dx*dx + dy*dy)
            angle = math.degrees(math.atan2(dy, dx)) % 360
            
            line_info = {
                "entity_id": line.entity_id,
                "start": (round(line.start_point.x, 3), round(line.start_point.y, 3)),
                "end": (round(line.end_point.x, 3), round(line.end_point.y, 3)),
                "length": round(length, 3),
                "angle": round(angle, 1),
                "layer": line.layer,
                "handle": line.handle
            }
            analysis["lines"].append(line_info)
    
    return analysis


def find_connected_components(entities, tolerance=0.5) -> List[List]:
    """Find connected geometry components using geometric tolerance."""
    components = []
    used_entities = set()
    
    def get_entity_points(entity):
        """Get key connection points from an entity."""
        points = []
        if entity.center:
            points.append((entity.center.x, entity.center.y))
        if entity.start_point:
            points.append((entity.start_point.x, entity.start_point.y))
        if entity.end_point:
            points.append((entity.end_point.x, entity.end_point.y))
        return points
    
    def points_connected(p1, p2):
        """Check if two points are within tolerance."""
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) <= tolerance
    
    def entities_connected(e1, e2):
        """Check if two entities are geometrically connected."""
        points1 = get_entity_points(e1)
        points2 = get_entity_points(e2)
        
        for p1 in points1:
            for p2 in points2:
                if points_connected(p1, p2):
                    return True
        return False
    
    # Build connected components
    for entity in entities:
        if entity.entity_id in used_entities:
            continue
            
        # Start new component
        component = [entity]
        used_entities.add(entity.entity_id)
        
        # Find all connected entities
        changed = True
        while changed:
            changed = False
            for other in entities:
                if other.entity_id in used_entities:
                    continue
                    
                # Check if connected to any entity in current component
                for comp_entity in component:
                    if entities_connected(comp_entity, other):
                        component.append(other)
                        used_entities.add(other.entity_id)
                        changed = True
                        break
        
        components.append(component)
    
    return components


def analyze_arc_reconstruction_potential(arc_analysis) -> Dict:
    """Analyze potential for reconstructing circles from arc groups."""
    reconstruction_analysis = {
        "potential_circles": [],
        "arc_groups_by_center": {}
    }
    
    # Analyze each center location
    for center_key, arcs in arc_analysis["by_center"].items():
        if len(arcs) < 2:
            continue  # Need at least 2 arcs
            
        # Group by compatible radius
        radius_groups = defaultdict(list)
        for arc in arcs:
            # Group similar radii by rounding to nearest 0.5
            radius_key = round(arc["radius"] * 2) / 2  # Round to nearest 0.5
            radius_groups[radius_key].append(arc)
        
        center_analysis = {
            "center": center_key,
            "arc_count": len(arcs),
            "radius_groups": {}
        }
        
        for radius, radius_arcs in radius_groups.items():
            if len(radius_arcs) < 2:
                continue
                
            # Calculate total coverage
            total_span = sum(arc["span"] for arc in radius_arcs)
            
            # Check for overlaps (simplified)
            angles = [(arc["start_angle"], arc["end_angle"]) for arc in radius_arcs]
            
            radius_analysis = {
                "radius": radius,
                "arc_count": len(radius_arcs),
                "total_span": total_span,
                "coverage_fraction": total_span / 360.0,
                "arcs": radius_arcs,
                "potential_circle": total_span > 300  # >300 degrees suggests circle
            }
            
            center_analysis["radius_groups"][radius] = radius_analysis
            
            if radius_analysis["potential_circle"]:
                reconstruction_analysis["potential_circles"].append({
                    "center": center_key,
                    "radius": radius,
                    "coverage": radius_analysis["coverage_fraction"],
                    "arc_count": len(radius_arcs)
                })
        
        reconstruction_analysis["arc_groups_by_center"][center_key] = center_analysis
    
    return reconstruction_analysis


def find_closed_loops(entities, tolerance=0.5) -> List[Dict]:
    """
    Find closed loops formed by lines and arcs.
    
    Distinguishes between:
    - True enclosed loops (non-zero area)  
    - Zero-area connectivity cycles (just connection chains)
    """
    loops = []
    
    # Get line and arc entities
    line_arc_entities = [e for e in entities if e.entity_type in [EntityType.LINE, EntityType.ARC]]
    
    if len(line_arc_entities) < 3:  # Need at least 3 entities for a loop
        return loops
    
    def get_endpoints(entity):
        """Get start and end points of line/arc."""
        if entity.entity_type == EntityType.LINE:
            return entity.start_point, entity.end_point
        elif entity.entity_type == EntityType.ARC and entity.center and entity.radius:
            # Calculate arc endpoints
            start_rad = math.radians(entity.start_angle)
            end_rad = math.radians(entity.end_angle)
            
            start_pt = Point2D(
                entity.center.x + entity.radius * math.cos(start_rad),
                entity.center.y + entity.radius * math.sin(start_rad)
            )
            end_pt = Point2D(
                entity.center.x + entity.radius * math.cos(end_rad),
                entity.center.y + entity.radius * math.sin(end_rad)
            )
            return start_pt, end_pt
        return None, None
    
    def points_match(p1, p2, tol):
        """Check if points match within tolerance."""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2) <= tol
    
    def calculate_polygon_area(vertices):
        """Calculate area using shoelace formula - returns 0 for degenerate polygons."""
        if len(vertices) < 3:
            return 0.0
            
        area = 0.0
        n = len(vertices)
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        return abs(area) / 2.0
    
    # Build connectivity graph
    connections = defaultdict(list)
    entity_endpoints = {}
    
    for entity in line_arc_entities:
        start, end = get_endpoints(entity)
        if start and end:
            entity_endpoints[entity.entity_id] = (start, end)
            
            # Find connections to other entities
            for other in line_arc_entities:
                if other.entity_id == entity.entity_id:
                    continue
                    
                other_start, other_end = get_endpoints(other)
                if other_start and other_end:
                    # Check all endpoint combinations
                    if points_match(end, other_start, tolerance):
                        connections[entity.entity_id].append((other.entity_id, 'end_to_start'))
                    elif points_match(end, other_end, tolerance):
                        connections[entity.entity_id].append((other.entity_id, 'end_to_end'))
                    elif points_match(start, other_start, tolerance):
                        connections[entity.entity_id].append((other.entity_id, 'start_to_start'))
                    elif points_match(start, other_end, tolerance):
                        connections[entity.entity_id].append((other.entity_id, 'start_to_end'))
    
    # Find cycles in the connectivity graph
    def find_cycles_from_entity(start_entity, max_depth=10):
        """Find cycles starting from a specific entity."""
        cycles = []
        
        def dfs(current, path, visited_connections):
            if len(path) > max_depth:
                return
                
            if len(path) > 2 and current == start_entity:
                # Found a cycle
                cycles.append(path[:])
                return
            
            for next_entity, connection_type in connections.get(current, []):
                if (current, next_entity, connection_type) in visited_connections:
                    continue
                    
                if next_entity in path[:-1]:  # Avoid loops except back to start
                    if next_entity == start_entity and len(path) >= 3:
                        cycles.append(path + [next_entity])
                    continue
                
                visited_connections.add((current, next_entity, connection_type))
                path.append(next_entity)
                dfs(next_entity, path, visited_connections)
                path.pop()
                visited_connections.remove((current, next_entity, connection_type))
        
        dfs(start_entity, [start_entity], set())
        return cycles
    
    # Find all cycles and filter by area
    found_cycles = []
    for entity in line_arc_entities:
        cycles = find_cycles_from_entity(entity.entity_id)
        for cycle in cycles:
            # Normalize cycle (start with lexicographically smallest ID)
            min_idx = cycle.index(min(cycle[:-1]))  # Exclude last (duplicate of first)
            normalized = cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]]
            
            if normalized not in found_cycles:
                found_cycles.append(normalized)
    
    # Analyze each cycle for geometric properties
    for cycle_ids in found_cycles:
        if len(cycle_ids) < 4:  # Need at least 3 unique entities
            continue
            
        cycle_entities = []
        total_perimeter = 0
        vertices = []
        
        for i, entity_id in enumerate(cycle_ids[:-1]):  # Exclude duplicate last
            entity = next((e for e in line_arc_entities if e.entity_id == entity_id), None)
            if entity:
                cycle_entities.append(entity)
                
                # Add to perimeter and collect vertices
                if entity.entity_type == EntityType.LINE:
                    start, end = get_endpoints(entity)
                    if start and end:
                        length = math.sqrt((end.x - start.x)**2 + (end.y - start.y)**2)
                        total_perimeter += length
                        vertices.append((start.x, start.y))
                elif entity.entity_type == EntityType.ARC and entity.radius:
                    # Calculate arc length
                    span_degrees = abs(entity.end_angle - entity.start_angle)
                    if span_degrees > 180:  # Handle wraparound
                        span_degrees = 360 - span_degrees
                    arc_length = (span_degrees / 360) * 2 * math.pi * entity.radius
                    total_perimeter += arc_length
                    
                    # For arcs, approximate with start point
                    start, _ = get_endpoints(entity)
                    if start:
                        vertices.append((start.x, start.y))
        
        # Calculate geometric properties
        if vertices and len(vertices) >= 3:
            # Calculate area to distinguish real loops from connectivity cycles
            area = calculate_polygon_area(vertices)
            
            # Bounding box
            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]
            bbox = {
                "min_x": min(xs), "max_x": max(xs),
                "min_y": min(ys), "max_y": max(ys)
            }
            bbox["width"] = bbox["max_x"] - bbox["min_x"]
            bbox["height"] = bbox["max_y"] - bbox["min_y"]
            bbox["center_x"] = (bbox["min_x"] + bbox["max_x"]) / 2
            bbox["center_y"] = (bbox["min_y"] + bbox["max_y"]) / 2
            
            # Classify loop type based on area
            area_threshold = 1.0  # Minimum area for "enclosed" vs "connectivity"
            loop_type = "enclosed" if area > area_threshold else "connectivity"
            
            loop_info = {
                "entity_ids": cycle_ids[:-1],  # Remove duplicate
                "entity_count": len(cycle_entities),
                "perimeter": round(total_perimeter, 3),
                "area": round(area, 3),
                "loop_type": loop_type,  # NEW: distinguish loop types
                "bounding_box": bbox,
                "vertices": [(round(x, 3), round(y, 3)) for x, y in vertices],
                "aspect_ratio": bbox["width"] / bbox["height"] if bbox["height"] > 0 else float('inf')
            }
            
            loops.append(loop_info)
    
    return loops


def perform_geometry_audit(dxf_path: Path) -> Dict:
    """Perform complete geometry audit of a DXF file."""
    print(f"\n{'='*80}")
    print(f"GEOMETRY AUDIT: {dxf_path.name}")
    print(f"{'='*80}")
    
    # Parse DXF
    entities = parse_dxf(dxf_path)
    normalized = normalize_geometry(entities)
    
    audit_result = {
        "dxf_file": dxf_path.name,
        "raw_entities": {},
        "bounding_box": {},
        "circles": {},
        "arcs": {},
        "lines": {},
        "connected_components": [],
        "arc_reconstruction": {},
        "closed_loops": []
    }
    
    # A. Raw entity inventory
    print("\nA. RAW ENTITY INVENTORY")
    print("-" * 30)
    
    entity_counts = defaultdict(int)
    for entity in entities:
        entity_counts[entity.entity_type.value] += 1
    
    audit_result["raw_entities"] = dict(entity_counts)
    print(f"Total entities: {len(entities)}")
    for entity_type, count in sorted(entity_counts.items()):
        print(f"  {entity_type}: {count}")
    
    # B. Geometry bounds
    print("\nB. GEOMETRY BOUNDS")
    print("-" * 20)
    
    bounds = calculate_bounding_box(entities)
    audit_result["bounding_box"] = bounds
    
    print(f"Bounding box:")
    print(f"  X: {bounds['min_x']:.3f} to {bounds['max_x']:.3f} (width: {bounds['width']:.3f})")
    print(f"  Y: {bounds['min_y']:.3f} to {bounds['max_y']:.3f} (height: {bounds['height']:.3f})")
    print(f"  Center: ({bounds['center_x']:.3f}, {bounds['center_y']:.3f})")
    
    # C. Connected geometry
    print("\nC. CONNECTED GEOMETRY")
    print("-" * 25)
    
    components = find_connected_components(entities)
    audit_result["connected_components"] = [
        [e.entity_id for e in comp] for comp in components
    ]
    
    print(f"Connected components: {len(components)}")
    for i, comp in enumerate(components):
        types = [e.entity_type.value for e in comp]
        type_counts = defaultdict(int)
        for t in types:
            type_counts[t] += 1
        print(f"  Component {i+1}: {len(comp)} entities - {dict(type_counts)}")
    
    # D. Explicit circles
    print("\nD. EXPLICIT CIRCLES")
    print("-" * 20)
    
    circle_analysis = analyze_circles(entities)
    audit_result["circles"] = circle_analysis
    
    print(f"Circle count: {circle_analysis['count']}")
    if circle_analysis['circles']:
        print(f"Radius range: {circle_analysis['radius_stats']['min']:.3f} to {circle_analysis['radius_stats']['max']:.3f}")
        print(f"Average radius: {circle_analysis['radius_stats']['avg']:.3f}")
        
        print("\nCircle details:")
        for circle in circle_analysis['circles']:
            print(f"  {circle['entity_id']}: center={circle['center']}, radius={circle['radius']}, layer='{circle['layer']}'")
    
    # E. Arc reconstruction
    print("\nE. ARC RECONSTRUCTION")
    print("-" * 25)
    
    arc_analysis = analyze_arcs(entities)
    audit_result["arcs"] = arc_analysis
    
    reconstruction = analyze_arc_reconstruction_potential(arc_analysis)
    audit_result["arc_reconstruction"] = reconstruction
    
    print(f"Arc count: {arc_analysis['count']}")
    print(f"Potential reconstructed circles: {len(reconstruction['potential_circles'])}")
    
    for potential in reconstruction['potential_circles']:
        print(f"  Center {potential['center']}: radius={potential['radius']:.1f}, "
              f"coverage={potential['coverage']:.1%}, arcs={potential['arc_count']}")
    
    # F. Closed loops
    print("\nF. CLOSED LINE/ARC LOOPS")
    print("-" * 30)
    
    closed_loops = find_closed_loops(entities)
    audit_result["closed_loops"] = closed_loops
    
    # Separate by loop type
    enclosed_loops = [loop for loop in closed_loops if loop.get("loop_type") == "enclosed"]
    connectivity_cycles = [loop for loop in closed_loops if loop.get("loop_type") == "connectivity"]
    
    print(f"Total loops found: {len(closed_loops)}")
    print(f"  Enclosed loops (area > 1.0): {len(enclosed_loops)}")
    print(f"  Connectivity cycles (area ≤ 1.0): {len(connectivity_cycles)}")
    
    for i, loop in enumerate(enclosed_loops):
        print(f"  Enclosed Loop {i+1}: {loop['entity_count']} entities, "
              f"perimeter={loop['perimeter']:.1f}, area={loop['area']:.1f}")
        print(f"    Size: {loop['bounding_box']['width']:.1f} x {loop['bounding_box']['height']:.1f}, "
              f"aspect ratio: {loop['aspect_ratio']:.2f}")
    
    if connectivity_cycles:
        print(f"  (+ {len(connectivity_cycles)} connectivity cycles with minimal area)")
    
    return audit_result


def audit_all_dxfs():
    """Audit all DXF files in the project."""
    dxf_dir = Path("dxf")
    dxf_files = sorted(dxf_dir.glob("*.dxf"))
    
    print("COMPREHENSIVE GEOMETRY AUDIT")
    print("="*80)
    print("Generic geometric analysis - NO product-specific expectations")
    
    all_results = {}
    
    for dxf_file in dxf_files:
        try:
            result = perform_geometry_audit(dxf_file)
            all_results[dxf_file.name] = result
        except Exception as e:
            print(f"\nERROR auditing {dxf_file.name}: {e}")
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("COMPARATIVE SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'File':<20} {'Circles':<8} {'Arcs':<6} {'Lines':<6} {'Loops':<6} {'Recon':<6}")
    print("-" * 60)
    
    for filename, result in all_results.items():
        circles = result["circles"]["count"]
        arcs = result["arcs"]["count"] 
        lines = result["raw_entities"].get("LINE", 0)
        loops = len(result["closed_loops"])
        recon = len(result["arc_reconstruction"]["potential_circles"])
        
        print(f"{filename:<20} {circles:<8} {arcs:<6} {lines:<6} {loops:<6} {recon:<6}")
    
    return all_results


if __name__ == "__main__":
    audit_all_dxfs()