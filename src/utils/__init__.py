"""
LCR Utility Modules

This package provides utility functions and classes for:
- Hardware compatibility checking
- Calibration validation
- Environment verification
- Data format conversion
"""

from .hardware_compatibility import (
    HardwareManager,
    GPUCapabilities,
    GPUArchitecture,
    HardwareCompatibilityError,
)

from .calibration_validator import (
    CalibrationValidator,
    CalibrationReport,
    compute_ece,
    compute_mce,
)

__all__ = [
    "HardwareManager",
    "GPUCapabilities",
    "GPUArchitecture",
    "HardwareCompatibilityError",
    "CalibrationValidator",
    "CalibrationReport",
    "compute_ece",
    "compute_mce",
]
