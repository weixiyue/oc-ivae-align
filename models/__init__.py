"""Explicit model selection for training interfaces and the shared evaluator."""
from __future__ import annotations

from typing import Any
from torch import nn

from models.oc_ivae import OC_iVAE
from models.ocald import OCALD
from models.proposal_cnn import ProposalCNN
from models.residual import ResidualModel
from models.vae import build_cvae_model, build_vae_model


def build_model(raw_config: dict[str, Any], model_config: dict[str, Any]) -> nn.Module:
    """Build one explicitly supported model without imposing a common forward tuple.

    Args:
        raw_config: Nested experiment configuration with optional baseline_type.
        model_config: Model dimensions and decoder conditioning settings.
    Returns:
        Model with its own forward interface.
    """
    model_type = raw_config.get('model_type', 'oc_ivae')
    if model_type == 'ocald':
        return OCALD(model_config)
    if model_type == 'proposal_cnn':
        return ProposalCNN(model_config)
    if model_type == 'residual':
        return ResidualModel(model_config)
    if model_type != 'oc_ivae':
        raise ValueError(f'unknown model_type: {model_type}')
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
