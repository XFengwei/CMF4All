"""Relations between core catalogue quantities."""

from __future__ import annotations

import pandas as pd

from .survey import Survey


def mass_radius_relation(survey: Survey) -> pd.DataFrame:
    """Return valid mass-radius points for one survey."""

    return survey.mass_radius_table()[["mass_msun", "radius_au"]]
