"""Runtime guards for security issues in required ML dependencies."""

from __future__ import annotations

from functools import wraps
from typing import Any


_TRUSTED_LIGHTNING_INSTANTIATORS = {
    "lightning.pytorch.cli.instantiate_module",
    "pytorch_lightning.cli.instantiate_module",
}


def harden_lightning_checkpoint_loading() -> None:
    """Backport Lightning's `_instantiator` allowlist until a fixed release exists.

    CVE-2026-58659 permits a checkpoint to import an arbitrary callable through
    a hyperparameter. The upstream fix is commit d710d68 and is not yet present
    in Lightning 2.6.5.
    """
    module_names = (
        "lightning.pytorch.core.saving",
        "pytorch_lightning.core.saving",
    )
    for module_name in module_names:
        try:
            module = __import__(module_name, fromlist=["_load_state"])
        except ImportError:
            continue
        original = getattr(module, "_load_state", None)
        if original is None or getattr(original, "_coherex_hardened", False):
            continue

        @wraps(original)
        def safe_load_state(cls, checkpoint: dict[str, Any], strict=None, _original=original, **kwargs):
            hparam_key = getattr(cls, "CHECKPOINT_HYPER_PARAMS_KEY", "hyper_parameters")
            hyperparameters = checkpoint.get(hparam_key, {})
            path = hyperparameters.get("_instantiator") if isinstance(hyperparameters, dict) else None
            path = kwargs.get("_instantiator", path)
            if path is not None and path not in _TRUSTED_LIGHTNING_INSTANTIATORS:
                raise ValueError(f"Blocked untrusted Lightning checkpoint instantiator: {path!r}")
            return _original(cls, checkpoint, strict=strict, **kwargs)

        safe_load_state._coherex_hardened = True
        module._load_state = safe_load_state
