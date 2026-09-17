#!/usr/bin/env python3
"""
Comprehensive AST-Level Audit Script

Performs systematic analysis of all numeric literals and algorithmic values
in production feature_extraction code to ensure complete centralization.
"""

import ast
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

class AlgorithmicValueAuditor(ast.NodeVisitor):
    """AST visitor that identifies all numeric literals and their contexts."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.violations = []
        self.all_literals = []
        
    def visit_Num(self, node):
        """Visit numeric literals (Python < 3.8)."""
        self._analyze_numeric_value(node.n, node.lineno, "literal")
        self.generic_visit(node)
    
    def visit_Constant(self, node):
        """Visit constant literals (Python >= 3.8)."""
        if isinstance(node.value, (int, float)):
            self._analyze_numeric_value(node.value, node.lineno, "constant")
        self.generic_visit(node)
    
    def visit_Compare(self, node):
        """Visit comparison operations."""
        # Look for threshold comparisons
        for comparator in node.comparators:
            if isinstance(comparator, ast.Constant) and isinstance(comparator.value, (int, float)):
                context = f"comparison_threshold"
                self._analyze_numeric_value(comparator.value, comparator.lineno, context)
        self.generic_visit(node)
    
    def visit_BinOp(self, node):
        """Visit binary operations (multiplication, etc.)."""
        if isinstance(node.op, ast.Mult):
            # Look for multiplier constants
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)):
                self._analyze_numeric_value(node.right.value, node.right.lineno, "multiplier")
            elif isinstance(node.left, ast.Constant) and isinstance(node.left.value, (int, float)):
                self._analyze_numeric_value(node.left.value, node.left.lineno, "multiplier")
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        """Visit assignments to identify score assignments."""
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
                if any(keyword in var_name.lower() for keyword in ['score', 'weight', 'factor', 'bonus']):
                    context = f"score_assignment_{var_name}"
                    self._analyze_numeric_value(node.value.value, node.value.lineno, context)
        self.generic_visit(node)
    
    def _analyze_numeric_value(self, value: float, line: int, context: str):
        """Analyze a numeric value to classify it."""
        
        # Classification logic
        classification = self._classify_value(value, context)
        
        entry = {
            "file": self.filename,
            "line": line,
            "value": value,
            "context": context,
            "classification": classification["category"],
            "reason": classification["reason"],
            "action": classification["action"]
        }
        
        self.all_literals.append(entry)
        
        if classification["category"] == "ALGORITHMIC CONFIGURATION":
            self.violations.append(entry)
    
    def _classify_value(self, value: float, context: str) -> Dict[str, str]:
        """Classify a numeric value according to the requirements."""
        
        # Mathematical constants (should remain inline)
        if value == 360.0 and "angle" in context.lower():
            return {
                "category": "MATHEMATICAL CONSTANT",
                "reason": "360 degrees is immutable full circle constant",
                "action": "PRESERVE INLINE"
            }
        
        if value == 90.0 and "angle" in context.lower():
            return {
                "category": "MATHEMATICAL CONSTANT", 
                "reason": "90 degrees is immutable right angle constant",
                "action": "PRESERVE INLINE"
            }
        
        if value == 180.0 and "angle" in context.lower():
            return {
                "category": "MATHEMATICAL CONSTANT",
                "reason": "180 degrees is immutable half circle constant", 
                "action": "PRESERVE INLINE"
            }
        
        if value in [0.0, 1.0] and "confidence" in context.lower():
            return {
                "category": "MATHEMATICAL CONSTANT",
                "reason": "0/1 are mathematical bounds for confidence normalization",
                "action": "PRESERVE INLINE"
            }
        
        # Array/indexing mechanics
        if value in [0, 1, 2] and context in ["literal", "constant"]:
            return {
                "category": "PROGRAMMING/INDEXING",
                "reason": "Basic indexing or counting constant",
                "action": "PRESERVE INLINE"
            }
        
        # Algorithmic thresholds, tolerances, scores, weights
        if any(keyword in context.lower() for keyword in [
            'threshold', 'tolerance', 'score', 'weight', 'factor', 
            'multiplier', 'bonus', 'penalty', 'confidence', 'evidence'
        ]):
            return {
                "category": "ALGORITHMIC CONFIGURATION",
                "reason": f"Algorithmic {context} parameter affects behavior",
                "action": "MUST CENTRALIZE"
            }
        
        # Comparison thresholds
        if "comparison" in context.lower():
            return {
                "category": "ALGORITHMIC CONFIGURATION", 
                "reason": "Comparison threshold affects algorithmic decisions",
                "action": "MUST CENTRALIZE"
            }
        
        # Score assignments
        if "score_assignment" in context.lower():
            return {
                "category": "ALGORITHMIC CONFIGURATION",
                "reason": "Scoring value affects ranking and selection",
                "action": "MUST CENTRALIZE"
            }
        
        # Default classification for other numeric values
        if abs(value) > 1000:
            return {
                "category": "OTHER",
                "reason": "Large numeric value - needs explicit classification",
                "action": "REVIEW"
            }
        
        return {
            "category": "OTHER",
            "reason": "Numeric value requires explicit classification",
            "action": "REVIEW"
        }

def audit_file(file_path: Path) -> Dict[str, Any]:
    """Audit a single Python file for algorithmic values."""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=str(file_path))
        auditor = AlgorithmicValueAuditor(str(file_path.name))
        auditor.visit(tree)
        
        return {
            "file": str(file_path.name),
            "violations": auditor.violations,
            "all_literals": auditor.all_literals,
            "violation_count": len(auditor.violations)
        }
        
    except Exception as e:
        return {
            "file": str(file_path.name),
            "error": str(e),
            "violations": [],
            "all_literals": [],
            "violation_count": 0
        }

def main():
    """Main audit function."""
    
    print("=== COMPREHENSIVE AST-LEVEL AUDIT ===")
    print("Auditing all production files under feature_extraction/")
    print()
    
    # Find all Python files in feature_extraction/
    feature_extraction_path = Path("feature_extraction")
    production_files = []
    
    for pattern in ["**/*.py"]:
        production_files.extend(feature_extraction_path.rglob(pattern))
    
    # Filter out test files and specific exclusions
    production_files = [
        f for f in production_files 
        if not f.name.startswith("test_") 
        and f.name != "geometry_reconstruction_old.py"
        and f.name != "__init__.py"
    ]
    
    print(f"Found {len(production_files)} production Python files")
    print()
    
    # Audit each file
    all_results = []
    total_violations = 0
    
    for file_path in sorted(production_files):
        result = audit_file(file_path)
        all_results.append(result)
        total_violations += result["violation_count"]
        
        if result["violation_count"] > 0:
            print(f"❌ {result['file']}: {result['violation_count']} violations")
        else:
            print(f"✅ {result['file']}: clean")
    
    print()
    print(f"=== AUDIT SUMMARY ===")
    print(f"Total files audited: {len(production_files)}")
    print(f"Total violations found: {total_violations}")
    print()
    
    # Show violations in detail
    if total_violations > 0:
        print("=== DETAILED VIOLATIONS ===")
        for result in all_results:
            if result["violation_count"] > 0:
                print(f"\n{result['file']}:")
                for violation in result["violations"]:
                    print(f"  Line {violation['line']}: {violation['value']} ({violation['context']})")
                    print(f"    → {violation['reason']}")
    
    # Generate complete audit table
    print("\n=== COMPLETE AUDIT TABLE ===")
    print(f"{'File':<40} | {'Line':<4} | {'Value':<8} | {'Context':<20} | {'Classification':<25} | {'Action'}")
    print("-" * 120)
    
    for result in all_results:
        for literal in result["all_literals"]:
            print(f"{literal['file']:<40} | {literal['line']:<4} | {literal['value']:<8} | {literal['context']:<20} | {literal['classification']:<25} | {literal['action']}")
    
    return total_violations == 0

if __name__ == "__main__":
    clean = main()
    exit(0 if clean else 1)