#!/usr/bin/env python3
"""
Test single image to verify fixes work
"""

import sys
import logging
from pathlib import Path

# Add project to path
sys.path.append('.')

from test_corrected_phase2_pipeline import test_corrected_phase2_pipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def main():
    # Test with first image only
    image_path = Path("data/inputs/WhatsApp Image 2026-09-09 at 12.07.42 PM.jpeg")
    
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return
    
    logger.info("Testing single image with corrected pipeline...")
    result = test_corrected_phase2_pipeline(image_path)
    
    if result:
        logger.info(f"✅ Success: {result.inspection_status.value} (score: {result.quality_metrics.overall_quality_score:.3f})")
        
        # Check outputs
        outputs_dir = Path("outputs")
        if outputs_dir.exists():
            image_stem = image_path.stem
            phase2_dir = outputs_dir / image_stem / "phase_2"
            if phase2_dir.exists():
                files = list(phase2_dir.glob("*"))
                logger.info(f"Output files created: {len(files)} files in {phase2_dir}")
                for f in files:
                    logger.info(f"  - {f.name}")
        
    else:
        logger.error("❌ Test failed")

if __name__ == "__main__":
    main()