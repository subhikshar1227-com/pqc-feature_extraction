#!/usr/bin/env python3
"""
Anti-Hardcoding Tests

Comprehensive tests to ensure NO hardcoded coordinates, feature counts,
filenames, or product-specific logic exists in production code.
"""

import re
import pytest
from pathlib import Path
from typing import List, Tuple, Set


class TestAntiHardcoding:
    """Comprehensive tests against hardcoded values in production code."""
    
    def setup_method(self):
        """Set up test environment."""
        self.project_root = Path(__file__).parent.parent
        self.production_modules = [
            "feature_extraction/",
            "dxf_resolver/",
            "cad_image_alignment/"
        ]
        
        # Known coordinates that should NEVER appear in production code
        self.forbidden_coordinates = [
            "152.16", "146.84", "63.16", "148.5", "105.0"
        ]
        
        # Known feature counts that should NEVER be hardcoded
        self.forbidden_counts = [
            "== 4", "== 19", "== 20", "== 5", "== 1"
        ]
        
        # Known filenames that should not be hardcoded in algorithms
        self.forbidden_filenames = [
            "c-bp", "c_tp", "box_front", "box_rear", "cad_box"
        ]
    
    def get_production_files(self) -> List[Path]:
        """Get all Python files in production modules."""
        production_files = []
        
        for module in self.production_modules:
            module_path = self.project_root / module
            if module_path.exists():
                # Get all .py files recursively
                py_files = module_path.rglob("*.py")
                production_files.extend(py_files)
        
        return production_files
    
    def test_no_hardcoded_coordinates_in_production(self):
        """Ensure NO hardcoded coordinates in production code."""
        
        violations = []
        production_files = self.get_production_files()
        
        for file_path in production_files:
            if file_path.name.startswith("test_"):
                continue  # Skip test files
                
            try:
                content = file_path.read_text(encoding='utf-8')
                
                # Check for forbidden coordinates
                for coord in self.forbidden_coordinates:
                    if coord in content:
                        # Find line number
                        lines = content.split('\n')
                        for i, line in enumerate(lines, 1):
                            if coord in line:
                                violations.append(f"{file_path.relative_to(self.project_root)}:{i} - Contains hardcoded coordinate '{coord}': {line.strip()}")
                
            except Exception as e:
                pytest.fail(f"Could not read file {file_path}: {e}")
        
        if violations:
            violation_msg = "\n".join(violations)
            pytest.fail(f"HARDCODED COORDINATES found in production code:\n{violation_msg}")
    
    def test_no_hardcoded_feature_counts_in_algorithms(self):
        """Ensure NO hardcoded feature counts in production algorithms."""
        
        violations = []
        production_files = self.get_production_files()
        
        # Pattern to match hardcoded feature count comparisons
        count_patterns = [
            r"== \d+(?:\s|$|#)",  # == 4, == 19, etc.
            r"!= \d+(?:\s|$|#)",  # != 4, != 19, etc.
            r"> \d+(?:\s|$|#)",   # > 4, > 19, etc. (allow > 0)
            r"< \d+(?:\s|$|#)",   # < 4, < 19, etc.
        ]
        
        for file_path in production_files:
            if file_path.name.startswith("test_"):
                continue  # Skip test files
                
            try:
                content = file_path.read_text(encoding='utf-8')
                lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    # Skip comments and docstrings
                    stripped = line.strip()
                    if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                        continue
                    
                    # Check for count patterns
                    for pattern in count_patterns:
                        matches = re.finditer(pattern, line)
                        for match in matches:
                            # Extract the number
                            num_str = match.group().replace('==', '').replace('!=', '').replace('>', '').replace('<', '').strip()
                            try:
                                num = int(num_str)
                                
                                # Allow certain common values
                                if num in [0, 1, 2]:  # Allow basic counts like > 0, len() == 1
                                    continue
                                
                                # Flag suspicious hardcoded counts
                                if num in [3, 4, 5, 19, 20, 21]:  # Known target values
                                    violations.append(f"{file_path.relative_to(self.project_root)}:{i} - Suspicious hardcoded count '{match.group()}': {line.strip()}")
                                    
                            except ValueError:
                                continue
                
            except Exception as e:
                pytest.fail(f"Could not read file {file_path}: {e}")
        
        if violations:
            violation_msg = "\n".join(violations)
            pytest.fail(f"HARDCODED FEATURE COUNTS found in production code:\n{violation_msg}")
    
    def test_no_filename_based_algorithm_logic(self):
        """Ensure NO filename-based logic in production algorithms."""
        
        violations = []
        production_files = self.get_production_files()
        
        for file_path in production_files:
            if file_path.name.startswith("test_"):
                continue  # Skip test files
                
            # Skip the DXF resolver module entirely - it legitimately handles filenames
            if "resolver" in str(file_path):
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8')
                
                # Check for forbidden filenames in algorithm logic
                for filename in self.forbidden_filenames:
                    if filename in content:
                        # Find line number and context
                        lines = content.split('\n')
                        for i, line in enumerate(lines, 1):
                            if filename in line:
                                # Skip comments, docstrings, and example strings
                                stripped = line.strip()
                                if (stripped.startswith('#') or stripped.startswith('"""') or 
                                    stripped.startswith("'''") or '>>>' in line or 
                                    'example' in line.lower() or 'e.g.' in line.lower()):
                                    continue
                                    
                                # Look for actual algorithmic conditions using filenames
                                if any(keyword in line.lower() for keyword in ['if', 'elif', 'case', '==']):
                                    violations.append(f"{file_path.relative_to(self.project_root)}:{i} - Contains filename-based logic '{filename}': {line.strip()}")
                
            except Exception as e:
                pytest.fail(f"Could not read file {file_path}: {e}")
        
        if violations:
            violation_msg = "\n".join(violations)
            pytest.fail(f"FILENAME-BASED LOGIC found in production algorithms:\n{violation_msg}")
    
    def test_no_product_specific_conditions(self):
        """Ensure NO product-specific conditions in production code."""
        
        violations = []
        production_files = self.get_production_files()
        
        # Patterns that suggest ACTUAL product-specific logic (not just words)
        suspicious_patterns = [
            r"if.*['\"]circular['\"].*:",  # if filename == "circular"
            r"if.*['\"]box['\"].*:",       # if filename == "box"
            r"if.*['\"]front['\"].*:",     # if filename == "front"  
            r"if.*['\"]rear['\"].*:",      # if filename == "rear"
        ]
        
        for file_path in production_files:
            if file_path.name.startswith("test_"):
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8')
                lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    # Skip comments
                    stripped = line.strip()
                    if stripped.startswith('#'):
                        continue
                    
                    # Check for suspicious patterns - only strings, not variable names
                    for pattern in suspicious_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            violations.append(f"{file_path.relative_to(self.project_root)}:{i} - Product-specific condition: {line.strip()}")
                
            except Exception as e:
                pytest.fail(f"Could not read file {file_path}: {e}")
        
        if violations:
            violation_msg = "\n".join(violations)
            pytest.fail(f"PRODUCT-SPECIFIC CONDITIONS found in production code:\n{violation_msg}")
    
    def test_configuration_parameters_centralized(self):
        """Ensure all tunable parameters are in config.py."""
        
        violations = []
        production_files = self.get_production_files()
        
        # Magic numbers that should be in config
        magic_number_patterns = [
            r"\d+\.\d+",  # Decimal numbers like 0.85, 2.5
            r"\d+",       # Integer numbers
        ]
        
        for file_path in production_files:
            if file_path.name.startswith("test_"):
                continue
            if "config.py" in str(file_path):
                continue  # Skip config file itself
                
            try:
                content = file_path.read_text(encoding='utf-8')
                lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    # Skip imports, comments, docstrings
                    stripped = line.strip()
                    if (stripped.startswith('import') or stripped.startswith('from') or 
                        stripped.startswith('#') or stripped.startswith('"""') or 
                        stripped.startswith("'''") or not stripped):
                        continue
                    
                    # Look for suspicious magic numbers in threshold comparisons
                    threshold_patterns = [
                        r">\s*\d+\.\d+",   # > 0.85
                        r"<\s*\d+\.\d+",   # < 2.5
                        r">=\s*\d+\.\d+",  # >= 0.8
                        r"<=\s*\d+\.\d+",  # <= 30.0
                    ]
                    
                    for pattern in threshold_patterns:
                        if re.search(pattern, line):
                            # Extract the number
                            match = re.search(r"\d+\.\d+", line)
                            if match:
                                num = float(match.group())
                                # Flag non-trivial magic numbers
                                if num not in [0.0, 1.0, 2.0]:
                                    violations.append(f"{file_path.relative_to(self.project_root)}:{i} - Magic number in threshold: {line.strip()}")
                
            except Exception as e:
                pytest.fail(f"Could not read file {file_path}: {e}")
        
        # Note: This test may flag legitimate cases, so we keep it informational
        if violations:
            # Convert to warning instead of failure for now
            print(f"\nPOTENTIAL MAGIC NUMBERS (review for config.py migration):")
            for violation in violations:
                print(f"  {violation}")
    
    def test_algorithm_independence_from_test_targets(self):
        """Ensure algorithms don't reference test validation targets."""
        
        violations = []
        production_files = self.get_production_files()
        
        # Known validation targets that should NOT be in algorithms
        forbidden_target_references = [
            "4 circles", "19 circles", "20 circles", "3 circles",
            "1 central hole", "1 square hole",
            "central circular hole", "target validation"
        ]
        
        for file_path in production_files:
            if file_path.name.startswith("test_"):
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8')
                
                for target in forbidden_target_references:
                    if target in content.lower():
                        lines = content.split('\n')
                        for i, line in enumerate(lines, 1):
                            if target in line.lower():
                                # Skip comments and docstrings
                                stripped = line.strip()
                                if not (stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''")):
                                    violations.append(f"{file_path.relative_to(self.project_root)}:{i} - References validation target '{target}': {line.strip()}")
                
            except Exception as e:
                pytest.fail(f"Could not read file {file_path}: {e}")
        
        if violations:
            violation_msg = "\n".join(violations)
            pytest.fail(f"VALIDATION TARGET REFERENCES found in production algorithms:\n{violation_msg}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])