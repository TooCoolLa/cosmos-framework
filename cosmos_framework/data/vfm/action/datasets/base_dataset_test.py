# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for ActionBaseDataset._build_result normalization handling."""

from pathlib import Path

import torch

from cosmos_framework.data.vfm.action.action_spec import Gripper, Joint, build_action_spec
from cosmos_framework.data.vfm.action.datasets.base_dataset import ActionBaseDataset


class _StubDataset(ActionBaseDataset):
    """Concrete subclass exposing _build_result without touching disk."""

    @property
    def action_dim(self) -> int:
        return 8

    def _action_spec(self):
        # 8D joint_pos layout: 7 arm joints + gripper (matches DROID joint_pos).
        return build_action_spec(Joint(n=7, label="arm"), Gripper())

    @classmethod
    def _stats_path(cls) -> Path:
        return Path("/nonexistent/stats.json")

    def __getitem__(self, idx):  # pragma: no cover - not exercised
        raise NotImplementedError


def _make_dataset(action_normalization, norm_stats=None) -> _StubDataset:
    # Bypass __init__ (which reads dataset files from disk) and set only the
    # attributes _build_result touches.
    ds = object.__new__(_StubDataset)
    ds._fps = 15.0
    ds._viewpoint = "concat_view"
    ds._domain_id = 0
    ds._action_normalization = action_normalization
    ds._norm_stats = norm_stats
    return ds


def _video() -> torch.Tensor:
    return torch.zeros(2, 3, 4, 4)  # [C, T, H, W] -> permuted inside _build_result


def test_build_result_skips_normalization_when_none():
    """action_normalization=None (raw joint_pos) must pass actions through unchanged."""
    action = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
    ds = _make_dataset(action_normalization=None)

    result = ds._build_result(mode="policy", video=_video(), action=action, ai_caption="x")

    assert torch.equal(result["action"], action)


def test_build_result_applies_normalization_when_method_set():
    """A configured method still normalizes (regression guard for the None fix)."""
    action = torch.full((4, 8), 0.5)
    stats = {"min": torch.zeros(8), "max": torch.ones(8)}
    ds = _make_dataset(action_normalization="minmax", norm_stats=stats)

    result = ds._build_result(mode="policy", video=_video(), action=action, ai_caption="x")

    # minmax with [0,1] range maps 0.5 -> 0.0; must differ from the raw input.
    assert torch.allclose(result["action"], torch.zeros(4, 8))
    assert not torch.equal(result["action"], action)
