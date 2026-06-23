from pathlib import Path

import joblib
import numpy as np
import torch

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # transformers depends on this, but keep the import defensive.
    hf_hub_download = None


class GazeFeatureTransformer:
    def __init__(
        self,
        transform="raw",
        raw_feature_dim=None,
        artifact_dir=None,
        artifact_repo_id=None,
        pca_components=2,
        gmm_components=5,
    ):
        self.transform = transform or "raw"
        self.raw_feature_dim = int(raw_feature_dim) if raw_feature_dim is not None else None
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.artifact_repo_id = artifact_repo_id
        self.pca_components = int(pca_components)
        self.gmm_components = int(gmm_components)
        self.scaler = None
        self.model = None

        if self.transform == "raw":
            self._output_dim = self.raw_feature_dim
            return

        self.scaler = joblib.load(self._resolve_artifact("gaze_scaler.joblib"))
        if self.transform == "pca":
            self.model = joblib.load(self._resolve_artifact(f"pca_{self.pca_components}.joblib"))
            self._output_dim = self.pca_components
        elif self.transform == "gmm":
            self.model = joblib.load(self._resolve_artifact(f"gmm_k{self.gmm_components}.joblib"))
            self._output_dim = self.gmm_components
        else:
            raise ValueError(f"Unknown gaze transform: {self.transform}")

    @property
    def output_dim(self):
        return self._output_dim

    def _resolve_artifact(self, filename):
        candidates = []
        if self.artifact_dir:
            candidates.extend([self.artifact_dir / filename, self.artifact_dir / "artifacts" / filename])
        if self.artifact_repo_id:
            repo_path = Path(str(self.artifact_repo_id))
            candidates.extend([repo_path / filename, repo_path / "artifacts" / filename])
        for candidate in candidates:
            if candidate.is_file():
                return candidate

        if self.artifact_repo_id and hf_hub_download is not None:
            try:
                return hf_hub_download(repo_id=self.artifact_repo_id, filename=filename)
            except Exception:
                return hf_hub_download(repo_id=self.artifact_repo_id, filename=f"artifacts/{filename}")

        searched = ", ".join(str(x) for x in candidates) or "<none>"
        raise FileNotFoundError(f"Could not resolve gaze artifact {filename}. Searched: {searched}")

    def transform_tensor(self, fixations, fixation_mask=None):
        if self.transform == "raw":
            return fixations

        device = fixations.device
        dtype = fixations.dtype
        shape = fixations.shape
        flat = fixations.detach().float().cpu().numpy().reshape(-1, shape[-1])
        out = np.zeros((flat.shape[0], self.output_dim), dtype=np.float32)

        if fixation_mask is None:
            valid = np.ones(flat.shape[0], dtype=bool)
        else:
            valid = fixation_mask.detach().cpu().numpy().reshape(-1).astype(bool)

        if valid.any():
            scaled = self.scaler.transform(flat[valid])
            if self.transform == "pca":
                out[valid] = self.model.transform(scaled).astype(np.float32)
            else:
                out[valid] = self.model.predict_proba(scaled).astype(np.float32)

        out = out.reshape(*shape[:-1], self.output_dim)
        return torch.as_tensor(out, dtype=dtype, device=device)
