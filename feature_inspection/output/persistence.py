"""
Output Persistence Manager

Handles saving Phase 2 results to structured output directories.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path
import time

from ..models.actual_feature import ActualFeatureSet
from ..models.feature_match import FeatureMatchSet
from ..models.inspection_result import InspectionResult
from .serialization import save_json

logger = logging.getLogger(__name__)


class OutputPersistence:
    """
    Manages structured saving of Phase 2 results.
    
    Creates organized output directories and saves all Phase 2 artifacts:
    - Actual features (original and transformed coordinates)
    - Matching results
    - Inspection results
    - Visualizations
    """
    
    def __init__(self, base_output_dir: Path = None):
        """
        Initialize output persistence manager.
        
        Args:
            base_output_dir: Base directory for outputs. Defaults to 'outputs/'
        """
        self.base_output_dir = base_output_dir or Path("outputs")
    
    def create_output_directory(self, image_path: Path) -> Path:
        """
        Create structured output directory for an image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Path to the created Phase 2 output directory
        """
        # Use image stem (filename without extension) for directory name
        image_stem = image_path.stem
        
        # Create directory structure: outputs/<image_stem>/phase_2/
        phase2_dir = self.base_output_dir / image_stem / "phase_2"
        phase2_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created output directory: {phase2_dir}")
        return phase2_dir
    
    def save_actual_features(self, 
                           actual_feature_set: ActualFeatureSet,
                           output_dir: Path,
                           coordinate_suffix: str = "") -> Path:
        """
        Save actual features to JSON.
        
        Args:
            actual_feature_set: Feature set to save
            output_dir: Directory to save in
            coordinate_suffix: Suffix to add to filename (e.g., "_transformed")
            
        Returns:
            Path to saved file
        """
        filename = f"actual_features{coordinate_suffix}.json"
        output_path = output_dir / filename
        
        save_json(actual_feature_set, output_path, "encode_actual_feature_set")
        
        logger.info(f"Saved actual features to {output_path}")
        return output_path
    
    def save_matching_results(self,
                            match_set: FeatureMatchSet,
                            output_dir: Path) -> Path:
        """
        Save feature matching results to JSON.
        
        Args:
            match_set: Feature match set to save
            output_dir: Directory to save in
            
        Returns:
            Path to saved file
        """
        output_path = output_dir / "matching_result.json"
        
        save_json(match_set, output_path, "encode_feature_match_set")
        
        logger.info(f"Saved matching results to {output_path}")
        return output_path
    
    def save_inspection_result(self,
                             inspection_result: InspectionResult, 
                             output_dir: Path) -> Path:
        """
        Save inspection result to JSON.
        
        Args:
            inspection_result: Inspection result to save
            output_dir: Directory to save in
            
        Returns:
            Path to saved file
        """
        output_path = output_dir / "inspection_result.json"
        
        save_json(inspection_result, output_path, "encode_inspection_result")
        
        logger.info(f"Saved inspection result to {output_path}")
        return output_path
    
    def save_transformation_metadata(self,
                                   transformation_metadata: Dict[str, Any],
                                   output_dir: Path) -> Path:
        """
        Save coordinate transformation metadata.
        
        Args:
            transformation_metadata: Transformation metadata dictionary
            output_dir: Directory to save in
            
        Returns:
            Path to saved file
        """
        output_path = output_dir / "coordinate_transformation.json"
        
        # Add timestamp
        metadata_with_timestamp = transformation_metadata.copy()
        metadata_with_timestamp["save_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(metadata_with_timestamp, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved transformation metadata to {output_path}")
        return output_path
    
    def save_all_phase2_outputs(self,
                              image_path: Path,
                              actual_features: ActualFeatureSet,
                              transformed_features: Optional[ActualFeatureSet],
                              match_set: FeatureMatchSet,
                              inspection_result: InspectionResult,
                              transformation_metadata: Optional[Dict[str, Any]] = None,
                              visualization_path: Optional[Path] = None) -> Dict[str, Path]:
        """
        Save complete Phase 2 output suite for an image.
        
        Args:
            image_path: Original input image path
            actual_features: Detected features in image coordinates
            transformed_features: Features transformed to DXF coordinates (if available)
            match_set: Feature matching results
            inspection_result: Quality inspection results
            transformation_metadata: Coordinate transformation metadata
            visualization_path: Path to generated visualization (if available)
            
        Returns:
            Dictionary mapping output type to saved file path
        """
        logger.info(f"Saving complete Phase 2 output suite for {image_path.name}")
        
        # Create output directory
        output_dir = self.create_output_directory(image_path)
        
        # CRITICAL: Clean up stale transformed features file if transformation is unavailable
        stale_transformed_file = output_dir / "actual_features_transformed.json"
        if stale_transformed_file.exists() and transformed_features is None:
            logger.info(f"Removing stale transformed features file: {stale_transformed_file}")
            stale_transformed_file.unlink()
        
        saved_paths = {}
        
        # Save actual features (original coordinates)
        saved_paths["actual_features"] = self.save_actual_features(
            actual_features, output_dir, ""
        )
        
        # Save transformed features ONLY if transformation succeeded
        if transformed_features is not None:
            # Double-check that transformed features are actually in DXF coordinates
            if transformed_features.coordinate_system == "dxf_mm":
                saved_paths["transformed_actual_features"] = self.save_actual_features(
                    transformed_features, output_dir, "_transformed"
                )
            else:
                logger.warning(f"Transformed features not in dxf_mm coordinate system: {transformed_features.coordinate_system}")
                logger.warning("Not saving transformed features file")
        
        # Save matching results
        saved_paths["matching_result"] = self.save_matching_results(
            match_set, output_dir
        )
        
        # Save inspection result  
        saved_paths["inspection_result"] = self.save_inspection_result(
            inspection_result, output_dir
        )
        
        # Save transformation metadata if available
        if transformation_metadata is not None:
            saved_paths["transformation_metadata"] = self.save_transformation_metadata(
                transformation_metadata, output_dir
            )
        
        # Copy visualization if provided
        if visualization_path is not None and visualization_path.exists():
            viz_output_path = output_dir / "actual_features.png"
            import shutil
            shutil.copy2(visualization_path, viz_output_path)
            saved_paths["visualization"] = viz_output_path
            logger.info(f"Copied visualization to {viz_output_path}")
        
        # Create summary file
        summary_path = self._create_output_summary(output_dir, saved_paths, image_path, inspection_result, transformation_metadata)
        saved_paths["summary"] = summary_path
        
        logger.info(f"Phase 2 output suite saved to {output_dir}")
        logger.info(f"Files saved: {list(saved_paths.keys())}")
        
        return saved_paths
    
    def _create_output_summary(self, 
                             output_dir: Path, 
                             saved_paths: Dict[str, Path],
                             image_path: Path,
                             inspection_result: InspectionResult,
                             transformation_metadata: Optional[Dict[str, Any]] = None) -> Path:
        """
        Create a truthful summary file for the output directory.
        
        Args:
            output_dir: Output directory
            saved_paths: Dictionary of saved file paths
            image_path: Original image path
            inspection_result: Inspection result for summary data
            transformation_metadata: Transformation metadata for accurate state reporting
            
        Returns:
            Path to summary file
        """
        # Determine actual pipeline state
        transformation_successful = (transformation_metadata and 
                                   transformation_metadata.get("transformation_successful", False))
        
        matching_performed = len(inspection_result.feature_match_set.matches) > 0 or (
            hasattr(inspection_result.feature_match_set, 'matching_algorithm') and
            inspection_result.feature_match_set.matching_algorithm != "blocked_coordinate_mismatch"
        )
        
        # Create truthful coordinate system summary
        coordinate_systems = {
            "expected_features": "dxf_mm",
            "actual_features_original": inspection_result.actual_feature_set.coordinate_system,
        }
        
        if transformation_successful:
            coordinate_systems["actual_features_transformed"] = "dxf_mm"
            coordinate_systems["matching_performed_in"] = "dxf_mm"
        else:
            coordinate_systems["transformation_status"] = "unavailable"
            if matching_performed:
                coordinate_systems["matching_performed_in"] = "mixed_coordinates"
            else:
                coordinate_systems["matching_status"] = "not_performed"
        
        summary_data = {
            "phase2_output_summary": {
                "input_image": str(image_path),
                "output_directory": str(output_dir),
                "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                
                "pipeline_state": {
                    "detection": "completed",
                    "transformation": "successful" if transformation_successful else "unavailable",
                    "matching": "performed" if matching_performed else "not_performed",
                    "comparison": "performed" if matching_performed else "unavailable",
                    "physical_quality": "determined" if matching_performed else "not_determined"
                },
                
                "inspection_summary": {
                    "status": inspection_result.inspection_status.value,
                    "quality_score": float(inspection_result.quality_metrics.overall_quality_score) if inspection_result.quality_metrics.overall_quality_score is not None else None,
                    "total_expected": inspection_result.inspection_summary.total_expected_features,
                    "total_actual": inspection_result.inspection_summary.total_actual_features,
                },
                
                "output_files": {
                    name: str(path.relative_to(output_dir)) for name, path in saved_paths.items()
                },
                
                "coordinate_systems": coordinate_systems
            }
        }
        
        # Only include matching statistics if matching was actually performed
        if matching_performed:
            summary_data["phase2_output_summary"]["inspection_summary"].update({
                "matched": inspection_result.inspection_summary.matched_features,
                "missing": inspection_result.inspection_summary.missing_features,
                "extra": inspection_result.inspection_summary.extra_features
            })
        else:
            summary_data["phase2_output_summary"]["inspection_summary"].update({
                "note": "Geometric matching not performed due to coordinate system incompatibility"
            })
        
        summary_path = output_dir / "output_summary.json"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        return summary_path