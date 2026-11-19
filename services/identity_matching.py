"""Global identity matching across all observations of a job.

Algorithm (all thresholds in ``UniqueCountConfig``):

1. Pairwise face similarity S_f (cosine of ArcFace embeddings).
2. Re-ID fusion -> fused similarity S:
     * S_f >= face_strong            : S = S_f (body never overrides strong face evidence)
     * face_weak <= S_f < face_strong: S = S_f + clip(w * (S_body - neutral), +-max_adjust)
                                       when both people have a reliable body embedding
     * S_f <  face_weak              : S = S_f (similar clothing alone never proves identity)
   For low-quality faces the weak bound is relaxed slightly so Re-ID can help
   with profile / occluded / badly lit faces.
3. Cannot-link: two faces from the same image are never the same person
   (S = -1, distance = max).
4. Agglomerative clustering, *average* linkage on 1 - S. Average linkage
   requires the mean similarity over all member pairs to pass the threshold,
   which prevents the transitive chaining of single linkage (A~B, B~C => A~C).
5. Cluster validation: split any remaining same-image conflicts and eject
   members whose mean similarity to the rest of the cluster is too low.
6. Ejected observations are re-matched against the validated clusters with
   the same average-similarity criterion; failures become their own identity
   and, if they were close, are reported as uncertain for manual review.
7. Body-only observations (no usable face) attach to an identity only with a
   strong OSNet match and a clear margin over the runner-up.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from services.config import UniqueCountConfig
from services.embedding_manager import EmbeddingManager, Observation
from utils.similarity import l2_normalize

logger = logging.getLogger(__name__)


@dataclass
class Identity:
    face_members: list[Observation] = field(default_factory=list)
    body_members: list[Observation] = field(default_factory=list)
    confidence: float = 0.0
    cohesion: float | None = None          # mean fused similarity between members
    nearest_other: float | None = None     # best similarity to any other identity
    match_basis: str = "face"
    representative: np.ndarray | None = None  # quality-weighted mean face embedding

    @property
    def observations(self) -> list[Observation]:
        return self.face_members + self.body_members

    @property
    def image_ids(self) -> list[str]:
        seen: dict[str, int] = {}
        for o in self.observations:
            seen.setdefault(o.image_id, o.image_index)
        return sorted(seen, key=seen.get)

    @property
    def first_seen(self) -> tuple[int, int]:
        return min((o.image_index, o.obs_id) for o in self.observations)


@dataclass
class MatchResult:
    identities: list[Identity]
    uncertain_matches: list[dict]
    unassigned_body_only: list[Observation]


class IdentityMatcher:
    def __init__(self, cfg: UniqueCountConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------ API
    def match(self, store: EmbeddingManager) -> MatchResult:
        face_obs = store.face_observations
        n = len(face_obs)
        logger.info("Identity matching: %d face observations, %d body-only observations",
                    n, len(store.body_only_observations))

        identities: list[Identity] = []
        uncertain: list[dict] = []
        if n:
            F = store.face_matrix[[o.face_row for o in face_obs]]
            img = np.array([o.image_index for o in face_obs])
            Q = np.array([o.face_quality for o in face_obs], dtype=np.float32)
            S_f = np.clip(F @ F.T, -1.0, 1.0)
            S_b, pair_b = self._body_similarity(store, face_obs)
            fused, reid_links = self._fuse(S_f, S_b, pair_b, Q)

            same_img = img[:, None] == img[None, :]
            np.fill_diagonal(same_img, False)
            fused[same_img] = -1.0
            np.fill_diagonal(fused, 1.0)

            clusters = self._cluster(fused, same_img)
            clusters, ejected = self._validate(clusters, fused, img, Q)
            clusters, rejected = self._reattach(clusters, ejected, fused, img, Q)
            logger.info("Clustering: %d identities (%d ejected during validation, %d re-attached)",
                        len(clusters), len(ejected), len(ejected) - len(rejected))

            labels = np.empty(n, dtype=int)
            for ci, members in enumerate(clusters):
                labels[members] = ci
            identities = self._build_identities(clusters, labels, face_obs, F, Q, fused, reid_links)
            if self.cfg.report_uncertain:
                uncertain = self._uncertain_pairs(labels, clusters, face_obs, img, S_f, S_b, pair_b, fused, Q, rejected)

        unassigned = self._attach_body_only(identities, store)
        return MatchResult(identities=identities, uncertain_matches=uncertain, unassigned_body_only=unassigned)

    # ------------------------------------------------------------ internals
    @staticmethod
    def _body_similarity(store: EmbeddingManager, face_obs: list[Observation]) -> tuple[np.ndarray, np.ndarray]:
        n = len(face_obs)
        has_b = np.array([o.body_row is not None and o.body_reliable for o in face_obs])
        if not has_b.any():
            return np.zeros((n, n), np.float32), np.zeros((n, n), bool)
        B = np.zeros((n, store.body_matrix.shape[1]), np.float32)
        B[has_b] = store.body_matrix[[o.body_row for o, h in zip(face_obs, has_b) if h]]
        return np.clip(B @ B.T, -1.0, 1.0), has_b[:, None] & has_b[None, :]

    def _fuse(self, S_f: np.ndarray, S_b: np.ndarray, pair_b: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        low_q = np.minimum(Q[:, None], Q[None, :]) < c.min_face_quality
        weak = np.where(low_q, c.face_weak_threshold - c.low_quality_weak_relax, c.face_weak_threshold)
        band = (S_f >= weak) & (S_f < c.face_strong_threshold) & pair_b
        adj = np.clip(c.body_weight * (S_b - c.body_neutral), -c.body_max_adjust, c.body_max_adjust)
        fused = S_f.astype(np.float32).copy()
        fused[band] += adj[band]
        # Links that pass the cluster threshold only thanks to Re-ID support.
        reid_links = band & (S_f < c.cluster_threshold) & (fused >= c.cluster_threshold)
        return fused, reid_links

    def _cluster(self, fused: np.ndarray, same_img: np.ndarray) -> list[list[int]]:
        n = fused.shape[0]
        if n == 1:
            return [[0]]
        D = np.clip(1.0 - fused, 0.0, 2.0)
        D[same_img] = 2.0
        np.fill_diagonal(D, 0.0)
        labels = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=1.0 - self.cfg.cluster_threshold,
        ).fit_predict(D)
        groups: dict[int, list[int]] = {}
        for i, lab in enumerate(labels):
            groups.setdefault(int(lab), []).append(i)
        return list(groups.values())

    def _validate(self, clusters: list[list[int]], fused: np.ndarray, img: np.ndarray, Q: np.ndarray) -> tuple[list[list[int]], list[int]]:
        valid, ejected = [], []
        for cluster in clusters:
            members = list(cluster)
            # 1) Same-image conflict: keep the member best supported by the rest.
            while True:
                uniq, counts = np.unique(img[members], return_counts=True)
                dup = uniq[counts > 1]
                if not len(dup):
                    break
                conflict = [m for m in members if img[m] == dup[0]]
                others = [m for m in members if img[m] != dup[0]]
                support = [fused[m, others].mean() if others else Q[m] for m in conflict]
                keep = conflict[int(np.argmax(support))]
                for m in conflict:
                    if m != keep:
                        members.remove(m)
                        ejected.append(m)
            # 2) Consistency: every member must agree with the cluster on average.
            while len(members) > 1:
                sub = fused[np.ix_(members, members)]
                mean_other = (sub.sum(axis=1) - 1.0) / (len(members) - 1)
                worst = int(np.argmin(mean_other))
                if mean_other[worst] >= self.cfg.member_min_similarity:
                    break
                ejected.append(members.pop(worst))
            valid.append(members)
        return valid, ejected

    def _reattach(self, clusters: list[list[int]], ejected: list[int], fused: np.ndarray,
                  img: np.ndarray, Q: np.ndarray) -> tuple[list[list[int]], dict[int, tuple[int, float]]]:
        """Re-match ejected observations; returns clusters and {obs: (best cluster, score)} for failures."""
        rejected: dict[int, tuple[int, float]] = {}
        for m in sorted(ejected, key=lambda x: -Q[x]):
            best, best_score = -1, -np.inf
            for ci, members in enumerate(clusters):
                if not members or (img[members] == img[m]).any():
                    continue
                s = float(fused[m, members].mean())
                if s > best_score:
                    best, best_score = ci, s
            if best >= 0 and best_score >= self.cfg.cluster_threshold:
                clusters[best].append(m)
            else:
                rejected[m] = (best, best_score)
                clusters.append([m])
        return [c for c in clusters if c], rejected

    def _build_identities(self, clusters, labels, face_obs, F, Q, fused, reid_links) -> list[Identity]:
        c = self.cfg
        other = fused.copy()
        other[labels[:, None] == labels[None, :]] = -1.0
        best_other_obs = other.max(axis=1) if len(face_obs) > 1 else np.full(len(face_obs), -1.0)

        identities = []
        for members in clusters:
            members = sorted(members)
            q = Q[members]
            top = np.argsort(-q)[: c.representative_top_k]
            rep = l2_normalize((F[members][top] * (q[top, None] + 1e-3)).sum(axis=0))

            nearest = float(best_other_obs[members].max())
            span = max(c.cluster_threshold - c.face_weak_threshold, 1e-6)
            separation = float(np.clip((c.cluster_threshold - nearest) / span, 0.0, 1.0))
            if len(members) > 1:
                sub = fused[np.ix_(members, members)]
                cohesion = float((sub.sum() - len(members)) / (len(members) * (len(members) - 1)))
                cohesion_conf = float(np.clip((cohesion - c.face_weak_threshold) / (0.75 - c.face_weak_threshold), 0, 1))
                basis = "face+reid" if reid_links[np.ix_(members, members)].any() else "face"
            else:
                cohesion, cohesion_conf, basis = None, float(q[0]), "single_observation"
            identities.append(Identity(
                face_members=[face_obs[m] for m in members],
                confidence=round(0.6 * cohesion_conf + 0.4 * separation, 3),
                cohesion=None if cohesion is None else round(cohesion, 4),
                nearest_other=round(nearest, 4) if nearest > -1 else None,
                match_basis=basis,
                representative=rep,
            ))
        return identities

    def _uncertain_pairs(self, labels, clusters, face_obs, img, S_f, S_b, pair_b, fused, Q, rejected) -> list[dict]:
        """Near-miss pairs of *different* identities that a human should review."""
        c = self.cfg
        cluster_imgs = [set(img[m].tolist()) for m in clusters]
        weak = c.face_weak_threshold - c.low_quality_weak_relax
        cand = (np.maximum(S_f, fused) >= weak) & (labels[:, None] != labels[None, :])
        cand &= img[:, None] != img[None, :]
        best: dict[tuple[int, int], tuple[float, int, int]] = {}
        for i, j in np.argwhere(np.triu(cand, 1)):
            a, b = sorted((int(labels[i]), int(labels[j])))
            if cluster_imgs[a] & cluster_imgs[b]:
                continue  # they co-occur in one image -> provably different people
            score = float(fused[i, j])
            if (a, b) not in best or score > best[(a, b)][0]:
                best[(a, b)] = (score, int(i), int(j))

        out = []
        for (a, b), (score, i, j) in sorted(best.items(), key=lambda kv: -kv[1][0])[: c.max_uncertain_reported]:
            reason = "ejected_from_cluster" if (i in rejected or j in rejected) else "similarity_below_threshold"
            out.append({
                "_identity_pair": (a, b),
                "observation_a": face_obs[i],
                "observation_b": face_obs[j],
                "face_similarity": round(float(S_f[i, j]), 4),
                "body_similarity": round(float(S_b[i, j]), 4) if pair_b[i, j] else None,
                "fused_similarity": round(score, 4),
                "min_face_quality": round(float(min(Q[i], Q[j])), 3),
                "reason": reason,
            })
        return out

    def _attach_body_only(self, identities: list[Identity], store: EmbeddingManager) -> list[Observation]:
        """Attach faceless person detections to identities on strong Re-ID evidence only."""
        c = self.cfg
        pending = [o for o in store.body_only_observations if o.body_reliable]
        unassigned = [o for o in store.body_only_observations if not o.body_reliable]
        if not pending:
            return unassigned
        B = store.body_matrix

        def best_matches(o: Observation, pool: list[Identity]) -> list[tuple[float, Identity]]:
            scores = []
            for ident in pool:
                if any(m.image_index == o.image_index for m in ident.observations):
                    continue
                rows = [m.body_row for m in ident.observations if m.body_row is not None and m.body_reliable]
                if rows:
                    scores.append((float((B[rows] @ B[o.body_row]).max()), ident))
            return sorted(scores, key=lambda t: -t[0])

        leftovers = []
        for o in pending:
            ranked = best_matches(o, identities)
            if ranked and ranked[0][0] >= c.body_only_attach_threshold and (
                len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= c.body_only_margin
            ):
                ranked[0][1].body_members.append(o)
            else:
                leftovers.append(o)

        if not c.count_body_only:
            return unassigned + leftovers

        # Optionally count unmatched faceless people, grouping them across
        # images with the same strict Re-ID rule.
        body_identities: list[Identity] = []
        for o in leftovers:
            ranked = best_matches(o, body_identities)
            if ranked and ranked[0][0] >= c.body_only_attach_threshold:
                ranked[0][1].body_members.append(o)
            else:
                body_identities.append(Identity(body_members=[o], confidence=0.3, match_basis="reid_only"))
        identities.extend(body_identities)
        return unassigned
