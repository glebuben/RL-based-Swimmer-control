"""
Advantage estimator package.

Factory
-------
make_advantage(name) parses a descriptor string and returns an Advantage instance.

Supported formats
-----------------
"QValue"                        -> QValueAdvantage()
"QBaseline_Exponential_<alpha>" -> QBaselineAdvantage(ExponentialBaseline(alpha))

Examples
--------
>>> make_advantage("QValue")
>>> make_advantage("QBaseline_Exponential_0.1")
>>> make_advantage(None)   # returns None
"""

from src.advantages.base import Advantage
from src.advantages.q_value import QValueAdvantage
from src.advantages.q_baseline import QBaselineAdvantage

__all__ = ["Advantage", "QValueAdvantage", "QBaselineAdvantage", "make_advantage"]


def make_advantage(name: str | None) -> Advantage | None:
    """
    Parse an advantage descriptor string and return the corresponding instance.

    Parameters
    ----------
    name : descriptor string, or None / "none" for no advantage estimator.

    Raises
    ------
    ValueError if the string is unrecognised or malformed.
    """
    if name is None or name.lower() == "none":
        return None

    parts = name.split("_", 1)
    kind  = parts[0].lower()

    if kind == "qvalue":
        return QValueAdvantage()

    if kind == "qbaseline":
        if len(parts) != 2:
            raise ValueError(
                f"QBaseline requires a baseline spec: 'QBaseline_<BaselineName>', got '{name}'"
            )
        from src.baselines import make_baseline
        baseline = make_baseline(parts[1])
        if baseline is None:
            raise ValueError(f"Could not parse baseline from '{parts[1]}'")
        return QBaselineAdvantage(baseline)

    raise ValueError(
        f"Unknown advantage '{name}'. "
        f"Supported: 'QValue', 'QBaseline_Exponential_<alpha>'"
    )