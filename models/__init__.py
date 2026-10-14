"""Explicit selection of the three existing model variants."""
from __future__ import annotations

from typing import Any

from models.oc_ivae import OC_iVAE
from models.vae import build_cvae_model, build_vae_model


def build_model(raw_config: dict[str, Any], model_config: dict[str, Any]) -> OC_iVAE:
    """Build the configured OC-iVAE, cVAE or VAE variant.

    Args:
        raw_config: Nested experiment configuration with optional baseline_type.
        model_config: Model dimensions and decoder conditioning settings.
    Returns:
        A model with the common twelve-field forward output.
    """
    baseline_type = raw_config.get("experiment", {}).get("baseline_type") or None
    if baseline_type is None:
        return OC_iVAE(model_config)
    if baseline_type == "cvae":
        return build_cvae_model(model_config)
    if baseline_type == "vae":
        return build_vae_model(model_config)
    raise ValueError(
        f"Unknown baseline_type {baseline_type!r}. "
        "Supported values: None, 'cvae', 'vae'."
    )
