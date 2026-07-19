"""
feature_definitions.py
----------------------

Feast feature repository definitions.

Feature Version 1
-----------------
Basic demographic and training history features.

Feature Version 2
-----------------
Version 1 plus engineered survey features and total_lift.

The feature engineering pipeline guarantees that all Version 2 feature
columns always exist in the parquet source, even when a category is not
present in the current dataset.
"""

from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float32, Int64

PARQUET_PATH = "data/feature_store/crossfit_features.parquet"

# ----------------------------------------------------------
# Entity
# ----------------------------------------------------------

athlete = Entity(
    name="athlete",
    join_keys=["athlete_id"],
    description="CrossFit athlete identifier",
)

# ----------------------------------------------------------
# Offline source
# ----------------------------------------------------------

crossfit_source = FileSource(
    name="crossfit_features_source",
    path=PARQUET_PATH,
    timestamp_field="event_timestamp",
)

# ----------------------------------------------------------
# Feature Version 1
# ----------------------------------------------------------

FEATURES_V1 = [
    Field(name="gender", dtype=Int64),
    Field(name="age", dtype=Float32),
    Field(name="height", dtype=Float32),
    Field(name="weight", dtype=Float32),
    Field(name="howlong", dtype=Int64),
]

crossfit_feature_view_v1 = FeatureView(
    name="crossfit_athlete_features_v1",
    entities=[athlete],
    ttl=timedelta(days=3650),
    schema=FEATURES_V1,
    source=crossfit_source,
    online=True,
    tags={
        "version": "v1",
        "description": "Baseline demographic features",
    },
)

# ----------------------------------------------------------
# Feature Version 2
# ----------------------------------------------------------

FEATURES_V2 = FEATURES_V1 + [
    Field(name="total_lift", dtype=Float32),

    Field(name="background__football", dtype=Int64),
    Field(name="background__weightlifting", dtype=Int64),

    Field(name="experience__beginner", dtype=Int64),
    Field(name="experience__advanced", dtype=Int64),

    Field(name="schedule__5_days", dtype=Int64),

    Field(name="eat__paleo", dtype=Int64),
]

crossfit_feature_view_v2 = FeatureView(
    name="crossfit_athlete_features_v2",
    entities=[athlete],
    ttl=timedelta(days=3650),
    schema=FEATURES_V2,
    source=crossfit_source,
    online=True,
    tags={
        "version": "v2",
        "description": "Baseline features plus engineered survey features",
    },
)

# ----------------------------------------------------------
# Feature Services
# ----------------------------------------------------------

crossfit_fs_v1 = FeatureService(
    name="crossfit_fs_v1",
    features=[crossfit_feature_view_v1],
    tags={"version": "v1"},
)

crossfit_fs_v2 = FeatureService(
    name="crossfit_fs_v2",
    features=[crossfit_feature_view_v2],
    tags={"version": "v2"},
)