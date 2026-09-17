"""
Feature Matching Engine

Main matching engine for pairing expected and actual features.
"""

import time
from typing import List, Dict, Any, Tuple, Optional
import logging

from feature_extraction.expected.feature_types import ExpectedFeature, ExpectedFeatureSet
from ..models.actual_feature import ActualFeature, ActualFeatureSet
from ..models.feature_match import FeatureMatch, FeatureMatchSet, MatchingStatistics, MatchType
from ..models.coordinate_transform import CoordinateTransform
from .geometric_matcher import GeometricMatcher
from ..config import (
    MATCH_ASSIGNMENT_METHOD, MATCH_MIN_CONFIDENCE,
    MATCH_CENTER_TOLERANCE_MM, MATCH_RADIUS_TOLERANCE_ABSOLUTE_MM, MATCH_RADIUS_TOLERANCE_RELATIVE,
    MATCH_DISTANCE_WEIGHT, MATCH_SIZE_WEIGHT, MATCH_TYPE_WEIGHT
)

logger = logging.getLogger(__name__)


class FeatureMatcher:
    """
    Main feature matching engine for pairing expected and actual features.
    
    Implements various matching algorithms and provides a unified interface
    for feature correspondence analysis.
    """
    
    def __init__(self, coordinate_transform: Optional[CoordinateTransform] = None):
        """
        Initialize feature matcher.
        
        Args:
            coordinate_transform: Transform between DXF and image coordinates
        """
        self.coordinate_transform = coordinate_transform
        self.geometric_matcher = GeometricMatcher(coordinate_transform)
        
        # Algorithm implementations
        self.assignment_algorithms = {
            "greedy": self._greedy_assignment,
            "hungarian": self._hungarian_assignment
        }
    
    def match_features(self, expected_set: ExpectedFeatureSet, 
                      actual_set: ActualFeatureSet) -> FeatureMatchSet:
        """
        Match expected features with actual features.
        
        Args:
            expected_set: Set of expected features from DXF analysis
            actual_set: Set of actual features from image detection
            
        Returns:
            FeatureMatchSet containing all matches and unmatched features
        """
        start_time = time.time()
        
        logger.info(f"Starting feature matching: {len(expected_set.features)} expected, "
                   f"{len(actual_set.features)} actual")
        
        # Calculate similarity matrix
        similarity_matrix, evidence_matrix = self._calculate_similarity_matrix(
            expected_set.features, actual_set.features
        )
        
        # Perform assignment using configured algorithm
        algorithm = MATCH_ASSIGNMENT_METHOD.lower()
        if algorithm not in self.assignment_algorithms:
            logger.warning(f"Unknown assignment algorithm '{algorithm}', using greedy")
            algorithm = "greedy"
        
        assignments = self.assignment_algorithms[algorithm](
            similarity_matrix, expected_set.features, actual_set.features
        )
        
        # Create feature matches
        matches, unmatched_expected, unmatched_actual = self._create_matches(
            assignments, expected_set.features, actual_set.features, 
            similarity_matrix, evidence_matrix
        )
        
        # Calculate statistics
        matching_time = time.time() - start_time
        statistics = self._calculate_matching_statistics(
            expected_set.features, actual_set.features, matches, matching_time
        )
        
        # Create match set
        match_set = FeatureMatchSet(
            matches=matches,
            unmatched_expected=unmatched_expected,
            unmatched_actual=unmatched_actual,
            matching_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            matching_statistics=statistics,
            matching_algorithm=algorithm,
            configuration_snapshot=self._get_configuration_snapshot()
        )
        
        logger.info(f"Feature matching complete: {len(matches)} matches, "
                   f"{len(unmatched_expected)} missing, {len(unmatched_actual)} extra "
                   f"(took {matching_time:.2f}s)")
        
        return match_set
    
    def _calculate_similarity_matrix(self, expected_features: List[ExpectedFeature],
                                   actual_features: List[ActualFeature]) -> Tuple[List[List[float]], List[List[Dict]]]:
        """Calculate similarity scores between all expected-actual feature pairs."""
        similarity_matrix = []
        evidence_matrix = []
        
        for i, expected in enumerate(expected_features):
            row_similarities = []
            row_evidence = []
            
            for j, actual in enumerate(actual_features):
                score, evidence = self.geometric_matcher.calculate_match_score(expected, actual)
                row_similarities.append(score)
                row_evidence.append(evidence)
            
            similarity_matrix.append(row_similarities)
            evidence_matrix.append(row_evidence)
        
        logger.debug(f"Calculated similarity matrix: {len(similarity_matrix)}x{len(similarity_matrix[0]) if similarity_matrix else 0}")
        return similarity_matrix, evidence_matrix
    
    def _greedy_assignment(self, similarity_matrix: List[List[float]],
                          expected_features: List[ExpectedFeature],
                          actual_features: List[ActualFeature]) -> List[Tuple[int, int, float]]:
        """Perform greedy assignment based on highest similarity scores."""
        if not similarity_matrix or not similarity_matrix[0]:
            return []
        
        assignments = []
        used_expected = set()
        used_actual = set()
        
        # Create list of all possible assignments with scores
        candidates = []
        for i in range(len(similarity_matrix)):
            for j in range(len(similarity_matrix[0])):
                score = similarity_matrix[i][j]
                if score >= MATCH_MIN_CONFIDENCE:
                    candidates.append((i, j, score))
        
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Greedily select non-conflicting assignments
        for expected_idx, actual_idx, score in candidates:
            if expected_idx not in used_expected and actual_idx not in used_actual:
                assignments.append((expected_idx, actual_idx, score))
                used_expected.add(expected_idx)
                used_actual.add(actual_idx)
        
        logger.debug(f"Greedy assignment: {len(assignments)} matches from {len(candidates)} candidates")
        return assignments
    
    def _hungarian_assignment(self, similarity_matrix: List[List[float]],
                            expected_features: List[ExpectedFeature],
                            actual_features: List[ActualFeature]) -> List[Tuple[int, int, float]]:
        """Perform optimal assignment using Hungarian algorithm."""
        try:
            from scipy.optimize import linear_sum_assignment
            import numpy as np
        except ImportError:
            logger.warning("scipy not available, falling back to greedy assignment")
            return self._greedy_assignment(similarity_matrix, expected_features, actual_features)
        
        if not similarity_matrix or not similarity_matrix[0]:
            return []
        
        # Convert similarity to cost matrix (Hungarian minimizes cost)
        cost_matrix = np.array(similarity_matrix)
        cost_matrix = 1.0 - cost_matrix  # Convert similarity to cost
        
        # Solve assignment problem
        expected_indices, actual_indices = linear_sum_assignment(cost_matrix)
        
        assignments = []
        for i in range(len(expected_indices)):
            expected_idx = expected_indices[i]
            actual_idx = actual_indices[i]
            similarity = similarity_matrix[expected_idx][actual_idx]
            
            # Only include assignments above minimum confidence
            if similarity >= MATCH_MIN_CONFIDENCE:
                assignments.append((expected_idx, actual_idx, similarity))
        
        logger.debug(f"Hungarian assignment: {len(assignments)} matches")
        return assignments
    
    def _create_matches(self, assignments: List[Tuple[int, int, float]],
                       expected_features: List[ExpectedFeature],
                       actual_features: List[ActualFeature],
                       similarity_matrix: List[List[float]],
                       evidence_matrix: List[List[Dict]]) -> Tuple[List[FeatureMatch], List[ExpectedFeature], List[ActualFeature]]:
        """Create FeatureMatch objects from assignments."""
        matches = []
        matched_expected_indices = set()
        matched_actual_indices = set()
        
        for expected_idx, actual_idx, match_score in assignments:
            expected = expected_features[expected_idx]
            actual = actual_features[actual_idx]
            
            # Create geometric comparison
            geometric_comparison = self.geometric_matcher.create_geometric_comparison(expected, actual)
            
            # Determine match type
            match_type = self.geometric_matcher.determine_match_type(match_score, geometric_comparison)
            
            # Get matching evidence
            match_evidence = evidence_matrix[expected_idx][actual_idx]
            
            # Create match
            feature_match = FeatureMatch(
                expected_feature=expected,
                actual_feature=actual,
                match_type=match_type,
                match_confidence=match_score,
                geometric_comparison=geometric_comparison,
                match_method="geometric_similarity",
                match_score=match_score,
                match_evidence=match_evidence
            )
            
            matches.append(feature_match)
            matched_expected_indices.add(expected_idx)
            matched_actual_indices.add(actual_idx)
        
        # Find unmatched features
        unmatched_expected = [
            expected_features[i] for i in range(len(expected_features))
            if i not in matched_expected_indices
        ]
        
        unmatched_actual = [
            actual_features[i] for i in range(len(actual_features))
            if i not in matched_actual_indices
        ]
        
        return matches, unmatched_expected, unmatched_actual
    
    def _calculate_matching_statistics(self, expected_features: List[ExpectedFeature],
                                     actual_features: List[ActualFeature],
                                     matches: List[FeatureMatch],
                                     matching_time: float) -> MatchingStatistics:
        """Calculate detailed matching statistics."""
        # Count matches by type
        exact_matches = len([m for m in matches if m.match_type == MatchType.EXACT_MATCH])
        good_matches = len([m for m in matches if m.match_type == MatchType.GOOD_MATCH])
        acceptable_matches = len([m for m in matches if m.match_type == MatchType.ACCEPTABLE_MATCH])
        poor_matches = len([m for m in matches if m.match_type == MatchType.POOR_MATCH])
        
        # Calculate averages
        total_matches = len(matches)
        avg_confidence = sum(m.match_confidence for m in matches) / max(1, total_matches)
        avg_accuracy = sum(m.geometric_comparison.overall_accuracy for m in matches) / max(1, total_matches)
        
        return MatchingStatistics(
            total_expected_features=len(expected_features),
            total_actual_features=len(actual_features),
            successful_matches=total_matches,
            exact_matches=exact_matches,
            good_matches=good_matches,
            acceptable_matches=acceptable_matches,
            poor_matches=poor_matches,
            unmatched_expected=len(expected_features) - total_matches,
            unmatched_actual=len(actual_features) - total_matches,
            average_match_confidence=avg_confidence,
            average_geometric_accuracy=avg_accuracy,
            matching_time_seconds=matching_time
        )
    
    def _get_configuration_snapshot(self) -> Dict[str, Any]:
        """Get current configuration for reproducibility."""
        return {
            "assignment_method": MATCH_ASSIGNMENT_METHOD,
            "min_confidence": MATCH_MIN_CONFIDENCE,
            "center_tolerance_mm": MATCH_CENTER_TOLERANCE_MM,
            "radius_tolerance_absolute_mm": MATCH_RADIUS_TOLERANCE_ABSOLUTE_MM,
            "radius_tolerance_relative": MATCH_RADIUS_TOLERANCE_RELATIVE,
            "distance_weight": MATCH_DISTANCE_WEIGHT,
            "size_weight": MATCH_SIZE_WEIGHT,
            "type_weight": MATCH_TYPE_WEIGHT
        }