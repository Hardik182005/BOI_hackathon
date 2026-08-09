"""The single model-ready frame builder.

Every training, tuning, calibration, explanation and scoring path in this
project obtains its feature matrix here. Nothing constructs a matrix by
itself, because doing so would bypass the Feature Availability Firewall.

The pipeline is fixed and short:

    raw canonical frame
      -> MetaFeatureBuilder  (adds the 13 interpretable MG_* domain features)
      -> firewall.admitted_features()  (removes everything not knowable
         at decision time, plus the fairness exclusions)
      -> firewall.assert_clean()  (belt and braces: refuses to return a
         matrix that contains a forbidden column even if the policy file
         were edited incorrectly)
      -> encode_dataframe()  (float32 matrix + frozen categorical maps)

`build_model_frame()` is deliberately cheap to call repeatedly - the raw
dataset, the registry and the meta-feature block are cached per process.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import polars as pl

from muleguard import settings
from muleguard.data import ingest
from muleguard.features import dictionary as fd
from muleguard.features import firewall
from muleguard.features.meta_features import MetaFeatureBuilder
from muleguard.features.preprocessing import encode_dataframe
from muleguard.logging import get_logger

log = get_logger("features.frame")


@dataclass
class ModelFrame:
    """A leakage-audited feature matrix plus everything needed to explain it."""

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    cat_maps: dict[str, list]
    decision: firewall.FirewallDecision
    view: str
    meta_features: list[str] = field(default_factory=list)
    n_rows: int = 0

    def summary(self) -> dict[str, Any]:
        d = self.decision.summary()
        d.update(
            view=self.view,
            n_rows=self.n_rows,
            n_features=len(self.feature_names),
            n_meta_features=len(self.meta_features),
            n_positives=int(self.y.sum()),
            prevalence=round(float(self.y.mean()), 6),
        )
        return d

    def subset(self, names: Sequence[str]) -> "ModelFrame":
        """Restrict to `names` (order follows the current matrix)."""
        wanted = set(names)
        missing = wanted - set(self.feature_names)
        if missing:
            raise KeyError(f"not in frame: {sorted(missing)[:8]}")
        idx = [i for i, n in enumerate(self.feature_names) if n in wanted]
        kept = [self.feature_names[i] for i in idx]
        return ModelFrame(
            X=self.X[:, idx],
            y=self.y,
            feature_names=kept,
            cat_maps={k: v for k, v in self.cat_maps.items() if k in wanted},
            decision=self.decision,
            view=self.view,
            meta_features=[m for m in self.meta_features if m in wanted],
            n_rows=self.n_rows,
        )


@functools.lru_cache(maxsize=1)
def _raw_frame() -> pl.DataFrame:
    return ingest.load_dataset()


@functools.lru_cache(maxsize=1)
def _registry() -> dict[str, Any]:
    return fd.load_registry()


@functools.lru_cache(maxsize=1)
def _augmented_frame() -> tuple[pl.DataFrame, tuple[str, ...], dict[str, Any]]:
    """Raw frame + MG_* meta-features, with the registry extended to match."""
    df = _raw_frame()
    reg = dict(_registry())
    builder = MetaFeatureBuilder(df.columns, registry=reg)
    out = builder.build(df)  # appends the MG_* block to a copy of `df`
    from muleguard.features.meta_features import attach_meta_to_registry

    reg = attach_meta_to_registry(reg)
    log.info("model frame: %d raw columns + %d meta-features",
             df.width, out.width - df.width)
    return out, tuple(builder.names()), reg


def raw_with_meta() -> pl.DataFrame:
    """The canonical dataset with the MG_* block appended (cached)."""
    return _augmented_frame()[0]


def meta_feature_names() -> list[str]:
    return list(_augmented_frame()[1])


def attach_meta(df: pl.DataFrame) -> pl.DataFrame:
    """Append the MG_* block to an arbitrary frame - typically an upload.

    The meta-features are row-wise functions of the raw columns, so they can be
    derived for data the model has never seen; nothing here consults the
    training set, and no fitted state is involved. A frame that already carries
    the block is returned unchanged, which keeps the call idempotent for
    callers that cannot tell whether it has been applied.
    """
    names = meta_feature_names()
    if names and all(n in df.columns for n in names):
        return df
    builder = MetaFeatureBuilder(df.columns, registry=dict(_registry()))
    return builder.build(df)


def augmented_registry() -> dict[str, Any]:
    return _augmented_frame()[2]


def build_model_frame(
    *,
    view: str | None = None,
    allow_classes: Iterable[str] | None = None,
    include_meta: bool = True,
    include_conditional: bool = False,
    include_gender: bool = False,
    include_demographics: bool = True,
    feature_subset: Sequence[str] | None = None,
    extra_allowed: Sequence[str] | None = None,
    cat_maps: dict[str, list] | None = None,
    df: pl.DataFrame | None = None,
) -> ModelFrame:
    """Build the model-ready matrix for one view.

    Args:
        view: named view from configs/feature_availability.yaml. ``None``
            means "all admissible classes", which is the default training pool.
        include_meta: append the MG_* domain meta-features.
        include_conditional: admit L1/L2/L3 risk flags - ONLY for a labelled
            ablation, never for a candidate model.
        include_gender: admit F3892 - ONLY for the fairness ablation.
        feature_subset: restrict to these names after firewall admission.
        extra_allowed: re-admit named quarantined columns. Every caller that
            passes this must label its results REJECTED LEAKAGE evidence.
    """
    frame = df if df is not None else raw_with_meta()
    reg = augmented_registry()
    metas = set(meta_feature_names())

    columns = [c for c in frame.columns if c != settings.TARGET_COLUMN]
    if not include_meta:
        columns = [c for c in columns if c not in metas]

    if view is not None:
        decision = firewall.view_features(view, columns, registry=reg)
    else:
        decision = firewall.admitted_features(
            columns,
            allow_classes=allow_classes or firewall.ADMISSIBLE_CLASSES,
            include_conditional=include_conditional,
            include_gender=include_gender,
            include_demographics=include_demographics,
            registry=reg,
        )

    admitted = list(decision.admitted)
    if extra_allowed:
        readmit = [c for c in extra_allowed if c in frame.columns and c not in set(admitted)]
        if readmit:
            log.warning(
                "ABLATION: re-admitting quarantined column(s) %s - results are "
                "REJECTED LEAKAGE evidence and must never be a candidate model",
                readmit,
            )
            admitted = admitted + readmit

    if feature_subset is not None:
        wanted = set(feature_subset)
        unknown = wanted - set(admitted)
        if unknown:
            raise ValueError(
                f"feature_subset contains columns the firewall did not admit: "
                f"{sorted(unknown)[:8]}"
            )
        admitted = [c for c in admitted if c in wanted]

    if not extra_allowed:
        firewall.assert_clean(admitted, context=f"model frame (view={view or 'ALL'})",
                              registry=reg)

    X, names, maps = encode_dataframe(frame, admitted, cat_maps=cat_maps)
    y = frame[settings.TARGET_COLUMN].cast(pl.Int32).to_numpy()
    return ModelFrame(
        X=X,
        y=y,
        feature_names=list(names),
        cat_maps=maps if cat_maps is None else dict(cat_maps),
        decision=decision,
        view=view or "ALL_ADMISSIBLE",
        meta_features=[n for n in names if n in metas],
        n_rows=frame.height,
    )


def refresh_quarantine_manifest() -> dict[str, Any]:
    """Rewrite artifacts/features/quarantined_features.json from the policy.

    Legacy code paths read that file through
    :func:`muleguard.features.preprocessing.load_quarantine_list`; keeping it
    in sync with configs/feature_availability.yaml means those paths inherit
    the firewall's decisions instead of the old four-column list.
    """
    from muleguard.utils import save_json

    frame = raw_with_meta()
    payload = firewall.quarantine_manifest(frame.columns, registry=augmented_registry())
    path = settings.FEATURES_DIR / "quarantined_features.json"
    save_json(payload, path)
    log.info("quarantine manifest refreshed: %d entries -> %s",
             len(payload["quarantine"]), path)
    return payload
