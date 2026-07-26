"""Artifact metrics and frozen confirmatory decisions."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np

from .data import AXES


BOOTSTRAP_SEED = 20260726
BOOTSTRAP_SAMPLES = 2000


def _vector(row: Mapping[str, Any], field: str = "utterance") -> np.ndarray:
    values = np.asarray(row[field], dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"{row.get('id', '?')} has an invalid {field} vector")
    return values


def _rounded(values: Sequence[float] | np.ndarray, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]


def cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    denominator = np.linalg.norm(left_array) * np.linalg.norm(right_array)
    if denominator <= 1e-12:
        return None
    return float(np.clip(np.dot(left_array, right_array) / denominator, -1.0, 1.0))


def clustered_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str = "utterance",
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("bootstrap rows cannot be empty")
    clusters: dict[str, list[np.ndarray]] = {}
    for row in rows:
        clusters.setdefault(str(row["case_id"]), []).append(_vector(row, field))
    cluster_ids = sorted(clusters)
    width = len(next(iter(clusters.values()))[0])
    rng = np.random.default_rng(seed)
    draws = np.empty((samples, width), dtype=np.float64)
    for sample_index in range(samples):
        selected = rng.integers(0, len(cluster_ids), size=len(cluster_ids))
        vectors = []
        for cluster_index in selected:
            options = clusters[cluster_ids[int(cluster_index)]]
            vectors.append(options[int(rng.integers(0, len(options)))])
        draws[sample_index] = np.mean(vectors, axis=0)
    observed = np.mean([_vector(row, field) for row in rows], axis=0)
    return {
        "mean": _rounded(observed),
        "lower": _rounded(np.quantile(draws, 0.025, axis=0)),
        "upper": _rounded(np.quantile(draws, 0.975, axis=0)),
        "clusters": len(cluster_ids),
        "observations": len(rows),
        "samples": samples,
        "seed": seed,
    }


def landmark_bootstrap(
    positive: Sequence[Mapping[str, Any]],
    negative: Sequence[Mapping[str, Any]],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if not positive or not negative:
        raise ValueError("landmark poles cannot be empty")
    positive_clusters: dict[str, list[np.ndarray]] = {}
    negative_clusters: dict[str, list[np.ndarray]] = {}
    for row in positive:
        positive_clusters.setdefault(str(row["case_id"]), []).append(_vector(row))
    for row in negative:
        negative_clusters.setdefault(str(row["case_id"]), []).append(_vector(row))
    positive_ids = sorted(positive_clusters)
    negative_ids = sorted(negative_clusters)
    width = len(next(iter(positive_clusters.values()))[0])
    rng = np.random.default_rng(seed)
    draws = np.empty((samples, width), dtype=np.float64)
    for sample_index in range(samples):
        positive_vectors = []
        negative_vectors = []
        for index in rng.integers(
            0,
            len(positive_ids),
            size=len(positive_ids),
        ):
            options = positive_clusters[positive_ids[int(index)]]
            positive_vectors.append(options[int(rng.integers(0, len(options)))])
        for index in rng.integers(
            0,
            len(negative_ids),
            size=len(negative_ids),
        ):
            options = negative_clusters[negative_ids[int(index)]]
            negative_vectors.append(options[int(rng.integers(0, len(options)))])
        draws[sample_index] = np.mean(positive_vectors, axis=0) - np.mean(
            negative_vectors, axis=0
        )
    observed = np.mean([_vector(row) for row in positive], axis=0) - np.mean(
        [_vector(row) for row in negative], axis=0
    )
    return {
        "mean": _rounded(observed),
        "lower": _rounded(np.quantile(draws, 0.025, axis=0)),
        "upper": _rounded(np.quantile(draws, 0.975, axis=0)),
        "positive_clusters": len(positive_ids),
        "negative_clusters": len(negative_ids),
        "observations": len(positive) + len(negative),
        "samples": samples,
        "seed": seed,
    }


def _centroid(
    rows: Sequence[Mapping[str, Any]], field: str = "utterance"
) -> np.ndarray:
    if not rows:
        raise ValueError("centroid rows cannot be empty")
    return np.mean([_vector(row, field) for row in rows], axis=0)


def _group_rows(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return grouped


def _landmark_label(group: str) -> str:
    axis_name, pole = group.split(":", maxsplit=1)
    axis = next(item for item in AXES if item.name == axis_name)
    return axis.positive_label if pole == "positive" else axis.negative_label


def summarize_landmarks(
    rows: Sequence[Mapping[str, Any]],
    axis_names: Sequence[str],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    landmark_rows = [row for row in rows if row["kind"] == "landmark"]
    axis_index = {name: index for index, name in enumerate(axis_names)}
    separations = []
    positive_target_means = 0
    positive_target_lowers = 0
    for axis in AXES:
        positive = [
            row
            for row in landmark_rows
            if row["axis"] == axis.name and row["pole"] == "positive"
        ]
        negative = [
            row
            for row in landmark_rows
            if row["axis"] == axis.name and row["pole"] == "negative"
        ]
        bootstrap = landmark_bootstrap(
            positive,
            negative,
            samples=bootstrap_samples,
            seed=BOOTSTRAP_SEED + axis_index[axis.name],
        )
        target = axis_index[axis.name]
        mean = np.asarray(bootstrap["mean"], dtype=np.float64)
        target_mean = float(mean[target])
        target_lower = float(bootstrap["lower"][target])
        positive_target_means += target_mean > 0
        positive_target_lowers += target_lower > 0
        vector_norm = float(np.linalg.norm(mean))
        separations.append(
            {
                "axis": axis.name,
                "positive_label": axis.positive_label,
                "negative_label": axis.negative_label,
                "target_index": target,
                "target_mean": round(target_mean, 6),
                "target_lower": round(target_lower, 6),
                "target_upper": round(float(bootstrap["upper"][target]), 6),
                "vector": bootstrap["mean"],
                "lower": bootstrap["lower"],
                "upper": bootstrap["upper"],
                "vector_norm": round(vector_norm, 6),
                "target_fraction": round(
                    abs(target_mean) / max(vector_norm, 1e-12),
                    6,
                ),
            }
        )
    return {
        "separations": separations,
        "positive_target_means": int(positive_target_means),
        "positive_target_lowers": int(positive_target_lowers),
        "orientation_pass": (
            positive_target_means == len(axis_names) and positive_target_lowers >= 10
        ),
    }


def summarize_atlas(
    rows: Sequence[Mapping[str, Any]],
    axis_names: Sequence[str],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    atlas_rows = [row for row in rows if row["kind"] == "atlas"]
    grouped = _group_rows(atlas_rows, "group")
    systems = sorted({str(row["system"]) for row in atlas_rows})
    families = []
    for family_index, (group, group_values) in enumerate(sorted(grouped.items())):
        bootstrap = clustered_bootstrap(
            group_values,
            samples=bootstrap_samples,
            seed=BOOTSTRAP_SEED + 100 + family_index,
        )
        by_system = {
            system: _rounded(
                _centroid([row for row in group_values if row["system"] == system])
            )
            for system in systems
        }
        pairwise = []
        for left, right in combinations(systems, 2):
            value = cosine(by_system[left], by_system[right])
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "cosine": None if value is None else round(value, 6),
                }
            )
        families.append(
            {
                "group": group,
                **bootstrap,
                "by_system": by_system,
                "template_cosines": pairwise,
            }
        )
    all_cosines = [
        pair["cosine"]
        for family in families
        for pair in family["template_cosines"]
        if pair["cosine"] is not None
    ]
    template_median = float(np.median(all_cosines))
    return {
        "axis_names": list(axis_names),
        "systems": systems,
        "families": families,
        "template_pairwise_cosine_median": round(template_median, 6),
        "template_stability_pass": template_median >= 0.80,
    }


def _landmark_centroids(
    rows: Sequence[Mapping[str, Any]],
    *,
    system: str | None = None,
    field: str = "utterance",
) -> dict[str, np.ndarray]:
    selected = [
        row
        for row in rows
        if row["kind"] == "landmark" and (system is None or row["system"] == system)
    ]
    return {
        _landmark_label(group): _centroid(values, field)
        for group, values in _group_rows(selected, "group").items()
    }


def nearest_centroids(
    vector: Sequence[float],
    centroids: Mapping[str, Sequence[float]],
) -> list[dict[str, Any]]:
    source = np.asarray(vector, dtype=np.float64)
    neighbors = []
    for name, values in centroids.items():
        target = np.asarray(values, dtype=np.float64)
        neighbors.append(
            {
                "group": name,
                "euclidean": round(float(np.linalg.norm(source - target)), 6),
                "cosine": (
                    None
                    if (similarity := cosine(source, target)) is None
                    else round(similarity, 6)
                ),
            }
        )
    return sorted(neighbors, key=lambda row: (row["euclidean"], row["group"]))


def summarize_meh(
    rows: Sequence[Mapping[str, Any]],
    atlas: Mapping[str, Any],
    axis_names: Sequence[str],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    atlas_rows = [row for row in rows if row["kind"] == "atlas"]
    meh_rows = [row for row in atlas_rows if row["group"] == "meh"]
    neutral_rows = [row for row in atlas_rows if row["group"] == "neutral"]
    bare_rows = [row for row in meh_rows if row["subset"] == "bare"]
    contextual_rows = [row for row in meh_rows if row["subset"] == "contextual"]
    meh_bootstrap = clustered_bootstrap(
        meh_rows,
        samples=bootstrap_samples,
        seed=BOOTSTRAP_SEED + 500,
    )
    neutral_bootstrap = clustered_bootstrap(
        neutral_rows,
        samples=bootstrap_samples,
        seed=BOOTSTRAP_SEED + 501,
    )
    bare_bootstrap = clustered_bootstrap(
        bare_rows,
        samples=bootstrap_samples,
        seed=BOOTSTRAP_SEED + 502,
    )
    contextual_bootstrap = clustered_bootstrap(
        contextual_rows,
        samples=bootstrap_samples,
        seed=BOOTSTRAP_SEED + 503,
    )
    meh_mean = np.asarray(meh_bootstrap["mean"], dtype=np.float64)
    neutral_mean = np.asarray(neutral_bootstrap["mean"], dtype=np.float64)
    delta = meh_mean - neutral_mean
    landmark_centroids = _landmark_centroids(rows)
    nearest = nearest_centroids(meh_mean, landmark_centroids)
    systems = atlas["systems"]
    nearest_by_system = {}
    boredom_systems = 0
    for system in systems:
        system_meh = _centroid([row for row in meh_rows if row["system"] == system])
        neighbors = nearest_centroids(
            system_meh,
            _landmark_centroids(rows, system=system),
        )
        nearest_by_system[system] = neighbors[:5]
        boredom_systems += neighbors[0]["group"] == "boredom"
    engagement_index = list(axis_names).index("engagement")
    non_null = (
        float(meh_bootstrap["upper"][engagement_index]) < 0
        and float(np.linalg.norm(delta)) >= 3.0
    )
    boredom_adjacent = nearest[0]["group"] == "boredom" and boredom_systems >= 2
    return {
        "mean": meh_bootstrap,
        "neutral": neutral_bootstrap,
        "meh_minus_neutral": _rounded(delta),
        "meh_minus_neutral_norm": round(float(np.linalg.norm(delta)), 6),
        "bare": bare_bootstrap,
        "contextual": contextual_bootstrap,
        "nearest_landmarks": nearest[:8],
        "nearest_landmarks_by_system": nearest_by_system,
        "boredom_nearest_systems": int(boredom_systems),
        "non_null_pass": bool(non_null),
        "boredom_adjacent_pass": bool(boredom_adjacent),
    }


def summarize_geometry(
    rows: Sequence[Mapping[str, Any]],
    atlas: Mapping[str, Any],
) -> dict[str, Any]:
    all_values = np.asarray([_vector(row) for row in rows], dtype=np.float64)
    correlation = np.corrcoef(all_values, rowvar=False)
    family_centroids = np.asarray(
        [family["mean"] for family in atlas["families"]],
        dtype=np.float64,
    )
    centered = family_centroids - family_centroids.mean(axis=0, keepdims=True)
    _u, singular_values, _vh = np.linalg.svd(centered, full_matrices=False)
    variances = singular_values**2
    first_fraction = float(variances[0] / max(variances.sum(), 1e-12))
    return {
        "axis_correlation": [_rounded(row, digits=6) for row in correlation],
        "first_component_variance_fraction": round(first_fraction, 6),
    }


def summarize_depth(
    rows: Sequence[Mapping[str, Any]],
    trace_layers: Sequence[int],
) -> dict[str, Any]:
    output = []
    for layer in trace_layers:
        field = f"__layer_{layer}"
        layer_rows = [
            {**row, field: row["utterance_by_layer"][str(layer)]} for row in rows
        ]
        layer_atlas = [row for row in layer_rows if row["kind"] == "atlas"]
        meh = _centroid(
            [row for row in layer_atlas if row["group"] == "meh"],
            field,
        )
        bare = _centroid(
            [
                row
                for row in layer_atlas
                if row["group"] == "meh" and row["subset"] == "bare"
            ],
            field,
        )
        contextual = _centroid(
            [
                row
                for row in layer_atlas
                if row["group"] == "meh" and row["subset"] == "contextual"
            ],
            field,
        )
        neutral = _centroid(
            [row for row in layer_atlas if row["group"] == "neutral"],
            field,
        )
        landmarks = _landmark_centroids(layer_rows, field=field)
        nearest = nearest_centroids(meh, landmarks)
        output.append(
            {
                "layer": int(layer),
                "meh": _rounded(meh),
                "bare": _rounded(bare),
                "contextual": _rounded(contextual),
                "neutral": _rounded(neutral),
                "meh_minus_neutral": _rounded(meh - neutral),
                "meh_minus_neutral_norm": round(
                    float(np.linalg.norm(meh - neutral)),
                    6,
                ),
                "nearest_landmark": nearest[0],
            }
        )
    return {"layers": output}


def summarize_measurements(
    rows: Sequence[Mapping[str, Any]],
    *,
    axis_names: Sequence[str],
    trace_layers: Sequence[int],
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("measurement rows cannot be empty")
    expected_width = len(axis_names)
    for row in rows:
        if len(row["utterance"]) != expected_width:
            raise ValueError(f"{row['id']} has the wrong coordinate width")
    landmarks = summarize_landmarks(
        rows,
        axis_names,
        bootstrap_samples=bootstrap_samples,
    )
    atlas = summarize_atlas(
        rows,
        axis_names,
        bootstrap_samples=bootstrap_samples,
    )
    meh = summarize_meh(
        rows,
        atlas,
        axis_names,
        bootstrap_samples=bootstrap_samples,
    )
    geometry = summarize_geometry(rows, atlas)
    depth = summarize_depth(rows, trace_layers)
    decisions = {
        "heldout_orientation": landmarks["orientation_pass"],
        "meh_non_null": meh["non_null_pass"],
        "meh_boredom_adjacent": meh["boredom_adjacent_pass"],
        "template_stability": atlas["template_stability_pass"],
    }
    return {
        "axis_names": list(axis_names),
        "rows": len(rows),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "landmarks": landmarks,
        "atlas": atlas,
        "meh": meh,
        "geometry": geometry,
        "depth": depth,
        "decisions": decisions,
    }


def compare_models(
    reference: Mapping[str, Any],
    extension: Mapping[str, Any],
) -> dict[str, Any]:
    reference_families = {
        row["group"]: row["mean"] for row in reference["atlas"]["families"]
    }
    extension_families = {
        row["group"]: row["mean"] for row in extension["atlas"]["families"]
    }
    common = sorted(set(reference_families) & set(extension_families))
    rows = []
    for group in common:
        left = np.asarray(reference_families[group], dtype=np.float64)
        right = np.asarray(extension_families[group], dtype=np.float64)
        similarity = cosine(left, right)
        sign_agreement = float(np.mean(np.sign(left) == np.sign(right)))
        rows.append(
            {
                "group": group,
                "cosine": (None if similarity is None else round(similarity, 6)),
                "sign_agreement": round(sign_agreement, 6),
            }
        )
    cosines = [row["cosine"] for row in rows if row["cosine"] is not None]
    meh = next(row for row in rows if row["group"] == "meh")
    return {
        "families": rows,
        "family_cosine_median": round(float(np.median(cosines)), 6),
        "family_sign_agreement_mean": round(
            float(np.mean([row["sign_agreement"] for row in rows])),
            6,
        ),
        "meh": meh,
    }
