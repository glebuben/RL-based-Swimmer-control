from src.baselines.base import Baseline
from src.baselines.exponential import ExponentialBaseline

__all__ = ["Baseline", "ExponentialBaseline", "make_baseline"]


def make_baseline(name: str | None) -> Baseline | None:
    """
    Parse a baseline descriptor string and return the corresponding instance.

    Supported formats
    -----------------
    None or "none"         -> no baseline
    "Exponential_<alpha>"  -> ExponentialBaseline(alpha=<alpha>)

    Examples
    --------
    >>> make_baseline("Exponential_0.1")
    ExponentialBaseline(alpha=0.1)
    >>> make_baseline(None)
    None
    """
    if name is None or name.lower() == "none":
        return None

    parts = name.split("_", 1)
    kind  = parts[0].lower()

    if kind == "exponential":
        if len(parts) != 2:
            raise ValueError(f"ExponentialBaseline requires alpha: 'Exponential_<alpha>', got '{name}'")
        alpha = float(parts[1])
        return ExponentialBaseline(alpha)

    raise ValueError(f"Unknown baseline '{name}'. Supported: 'Exponential_<alpha>'")