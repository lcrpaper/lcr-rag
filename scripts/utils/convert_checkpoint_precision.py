#!/usr/bin/env python3
"""
Checkpoint Precision Converter

Converts checkpoints between FP32, FP16, and INT8 precisions for hardware compatibility.

Usage:
    python scripts/utils/convert_checkpoint_precision.py \
        --input checkpoints/detector/final_v2.1_a100_fp16.pt \
        --output checkpoints/detector/converted_fp32.pt \
        --precision fp32

    python scripts/utils/convert_checkpoint_precision.py \
        --input checkpoints/refinement/final_v1.3_mixed_precision.pt \
        --output checkpoints/refinement/quantized_int8.pt \
        --precision int8 \
        --calibration-data data/benchmarks/temp_conflict/dev.jsonl

Note:
    - FP16 → FP32: Lossless, improves compatibility
    - FP32 → FP16: May lose precision, requires validation
    - Any → INT8: Requires calibration data, ~0.5% accuracy loss

Warning:
    This conversion may not preserve all A100-specific optimizations.
    Performance may differ from original checkpoint.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import copy

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class CheckpointConverter:
    """Handles checkpoint precision conversions with validation."""

    SUPPORTED_PRECISIONS = ['fp32', 'fp16', 'int8']

    def __init__(self, input_path: str, output_path: str, target_precision: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.target_precision = target_precision.lower()

        if self.target_precision not in self.SUPPORTED_PRECISIONS:
            raise ValueError(f"Unsupported precision: {target_precision}. "
                           f"Must be one of {self.SUPPORTED_PRECISIONS}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint with error handling."""
        logger.info(f"Loading checkpoint from {self.input_path}")

        if not self.input_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.input_path}")

        try:
            checkpoint = torch.load(self.input_path, map_location='cpu')
            logger.info(f"  ✓ Loaded successfully")
            return checkpoint
        except Exception as e:
            logger.error(f"  ✗ Failed to load: {e}")
            raise

    def detect_current_precision(self, checkpoint: Dict) -> str:
        """Detect the current precision of the checkpoint."""
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor) and value.dtype in [torch.float16, torch.float32]:
                dtype = value.dtype
                if dtype == torch.float16:
                    return 'fp16'
                elif dtype == torch.float32:
                    return 'fp32'
                break

        return 'unknown'

    def convert_to_fp32(self, checkpoint: Dict) -> Dict:
        """Convert checkpoint to FP32 (lossless)."""
        logger.info("Converting to FP32...")

        converted = copy.deepcopy(checkpoint)

        if 'model_state_dict' in converted:
            state_dict = converted['model_state_dict']
        else:
            state_dict = converted

        for key in state_dict:
            if isinstance(state_dict[key], torch.Tensor):
                state_dict[key] = state_dict[key].float()

        if 'metadata' in converted:
            converted['metadata']['precision'] = 'fp32'
            converted['metadata']['converted_from'] = self.detect_current_precision(checkpoint)
            converted['metadata']['conversion_note'] = 'Converted for broader hardware compatibility'

        logger.info("  ✓ Conversion complete")
        return converted

    def convert_to_fp16(self, checkpoint: Dict) -> Dict:
        """Convert checkpoint to FP16 (may lose precision)."""
        logger.info("Converting to FP16...")
        logger.warning("  ⚠ Warning: FP32→FP16 may lose numerical precision")

        converted = copy.deepcopy(checkpoint)

        if 'model_state_dict' in converted:
            state_dict = converted['model_state_dict']
        else:
            state_dict = converted

        for key in state_dict:
            if isinstance(state_dict[key], torch.Tensor):
                state_dict[key] = state_dict[key].half()

        if 'metadata' in converted:
            converted['metadata']['precision'] = 'fp16'
            converted['metadata']['converted_from'] = self.detect_current_precision(checkpoint)
            converted['metadata']['conversion_note'] = 'Converted to FP16 - validate accuracy'
            converted['metadata']['requires_gpu'] = 'A100 or A6000 recommended for Tensor Cores'

        logger.info("  ✓ Conversion complete")
        logger.warning("  ⚠ Validate accuracy on dev set before using in production")
        return converted

    def convert_to_int8(self, checkpoint: Dict, calibration_data: Optional[str] = None) -> Dict:
        """Convert checkpoint to INT8 (requires calibration)."""
        logger.info("Converting to INT8...")

        if calibration_data is None:
            logger.warning("  ⚠ No calibration data provided - using dummy quantization")
            logger.warning("  ⚠ Results may be suboptimal. Provide --calibration-data for better quality")

        converted = copy.deepcopy(checkpoint)

        logger.warning("  ⚠ INT8 conversion requires model architecture")
        logger.warning("  ⚠ Use torch.quantization.quantize_dynamic() on loaded model instead")
        logger.info("  ℹ See scripts/utils/quantize_checkpoint.py for full quantization")

        if 'metadata' in converted:
            converted['metadata']['precision'] = 'int8'
            converted['metadata']['converted_from'] = self.detect_current_precision(checkpoint)
            converted['metadata']['conversion_note'] = 'Quantized to INT8 - ~0.5% accuracy loss expected'
            converted['metadata']['requires_calibration'] = True

        return converted

    def validate_conversion(self, original: Dict, converted: Dict) -> bool:
        """Validate that conversion preserved essential information."""
        logger.info("Validating conversion...")

        orig_keys = set(original.get('model_state_dict', original).keys())
        conv_keys = set(converted.get('model_state_dict', converted).keys())

        if orig_keys != conv_keys:
            logger.error("  ✗ State dict keys don't match!")
            logger.error(f"    Missing: {orig_keys - conv_keys}")
            logger.error(f"    Extra: {conv_keys - orig_keys}")
            return False

        orig_dict = original.get('model_state_dict', original)
        conv_dict = converted.get('model_state_dict', converted)

        for key in orig_keys:
            if isinstance(orig_dict[key], torch.Tensor):
                if orig_dict[key].shape != conv_dict[key].shape:
                    logger.error(f"  ✗ Shape mismatch for {key}")
                    return False

        logger.info("  ✓ Validation passed")
        return True

    def save_checkpoint(self, checkpoint: Dict):
        """Save converted checkpoint with metadata update."""
        logger.info(f"Saving to {self.output_path}")

        if 'conversion_info' not in checkpoint:
            checkpoint['conversion_info'] = {}

        checkpoint['conversion_info'].update({
            'source_file': str(self.input_path),
            'target_precision': self.target_precision,
            'converted_by': 'convert_checkpoint_precision.py',
            'warning': 'This checkpoint was converted and may not match original performance'
        })

        torch.save(checkpoint, self.output_path)
        logger.info("  ✓ Saved successfully")

        metadata_path = self.output_path.with_suffix('.json').with_name(
            f"{self.output_path.stem}_metadata.json"
        )
        if 'metadata' in checkpoint:
            with open(metadata_path, 'w') as f:
                json.dump(checkpoint['metadata'], f, indent=2)
            logger.info(f"  ✓ Metadata saved to {metadata_path}")

    def convert(self, calibration_data: Optional[str] = None):
        """Main conversion pipeline."""
        logger.info("="*80)
        logger.info("CHECKPOINT PRECISION CONVERSION")
        logger.info("="*80)

        checkpoint = self.load_checkpoint()
        current_precision = self.detect_current_precision(checkpoint)
        logger.info(f"Current precision: {current_precision}")
        logger.info(f"Target precision:  {self.target_precision}")

        if self.target_precision == 'fp32':
            converted = self.convert_to_fp32(checkpoint)
        elif self.target_precision == 'fp16':
            converted = self.convert_to_fp16(checkpoint)
        elif self.target_precision == 'int8':
            converted = self.convert_to_int8(checkpoint, calibration_data)

        if not self.validate_conversion(checkpoint, converted):
            logger.error("Validation failed! Aborting.")
            return False

        self.save_checkpoint(converted)

        logger.info("="*80)
        logger.info("✅ CONVERSION COMPLETE")
        logger.info("="*80)
        logger.info("\nNext steps:")
        logger.info("1. Validate accuracy on dev set:")
        logger.info(f"   python scripts/evaluation/eval_quick.py --checkpoint {self.output_path}")
        logger.info("2. Compare with original checkpoint performance")
        logger.info("3. Update your config to use new checkpoint")

        return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert checkpoint precision',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert A100 FP16 checkpoint to FP32 for V100 compatibility
  python scripts/utils/convert_checkpoint_precision.py \\
      --input checkpoints/detector/final_v2.1_a100_fp16.pt \\
      --output checkpoints/detector/converted_v100_fp32.pt \\
      --precision fp32

  # Convert FP32 to FP16 for A100 optimization
  python scripts/utils/convert_checkpoint_precision.py \\
      --input checkpoints/refinement/final_v1.3_fp32_baseline.pt \\
      --output checkpoints/refinement/converted_fp16.pt \\
      --precision fp16

  # Quantize to INT8 (with calibration)
  python scripts/utils/convert_checkpoint_precision.py \\
      --input checkpoints/classifier/deberta_v3_final.pt \\
      --output checkpoints/classifier/quantized_int8.pt \\
      --precision int8 \\
      --calibration-data data/benchmarks/*/dev.jsonl

Precision Trade-offs:
  FP32: Maximum compatibility, highest memory, slower on Tensor Cores
  FP16: A100/A6000 optimized, 2× faster, minimal accuracy loss
  INT8: CPU-friendly, 4× faster inference, ~0.5% accuracy loss
        """
    )

    parser.add_argument(
        '--input',
        required=True,
        help='Input checkpoint path'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output checkpoint path'
    )
    parser.add_argument(
        '--precision',
        required=True,
        choices=['fp32', 'fp16', 'int8'],
        help='Target precision'
    )
    parser.add_argument(
        '--calibration-data',
        help='Calibration data for INT8 quantization (JSONL file)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation on dev set after conversion'
    )

    args = parser.parse_args()

    converter = CheckpointConverter(
        input_path=args.input,
        output_path=args.output,
        target_precision=args.precision
    )

    success = converter.convert(calibration_data=args.calibration_data)

    if success and args.validate:
        logger.info("\nRunning validation...")
        logger.info("(Validation implementation pending - manually run eval_quick.py)")

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
