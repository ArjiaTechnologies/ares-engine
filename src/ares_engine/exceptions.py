"""Domain-specific exceptions."""


class AresError(Exception):
    """Base exception for ARES Engine."""


class ConfigurationError(AresError):
    """Raised when configuration is invalid."""


class DataQualityError(AresError):
    """Raised when market data fails a hard quality gate."""


class BundleIntegrityError(AresError):
    """Raised when a model bundle is missing files or fails hash verification."""


class PromotionRejected(AresError):
    """Raised when a challenger does not satisfy promotion rules."""
