"""Assemble GET-Zero per-joint tokens: one token per degree of freedom.

GET-Zero (Patel and Song, ICRA 2025) builds token ``j`` at time ``t`` as

    T_j^t = [o_vg^{t-H+1..t}, o_fg, o_vl,j^{t-H+1..t}, o_fl,j]

* ``o_vg`` variable global observations with history, shared by every token
  (GET-Zero: a sinusoidal task phase; for a legged robot: base angular velocity,
  projected gravity, velocity command).
* ``o_fg`` fixed global observations (for example a task identifier).
* ``o_vl,j`` variable local observations of joint ``j`` with history (GET-Zero:
  the normalized joint position and the PD target position).
* ``o_fl,j`` fixed local hardware properties of joint ``j`` from URDF, given by
  ``EmbodimentGraph.joint_features``.

History is flattened feature-major, as in the GET-Zero code: all timesteps of the
first feature, then all timesteps of the second feature, and so on. Pass
history ordered oldest to newest. The encoder applies a layer norm and a learned
linear embedding to these raw tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DaGetTokenizer:
    """Concatenate global and per-joint observations into GET-Zero tokens.

    Attributes:
        fixed_local_size (int): Hardware features per joint, F_fl.
        variable_local_size (int): Per-joint observations per timestep, F_vl.
        variable_global_size (int): Global observations per timestep, F_vg.
        fixed_global_size (int): Constant global features, F_fg.
        history_size (int): Number of timesteps H of variable observations.
    """

    fixed_local_size: int
    variable_local_size: int = 0
    variable_global_size: int = 0
    fixed_global_size: int = 0
    history_size: int = 1

    def __post_init__(self) -> None:
        """Reject negative sizes and an empty token.

        Returns:
            None: The tokenizer is valid after construction.

        Raises:
            ValueError: A size is negative, history is not positive, or the
                token would have no features.
        """
        sizes = (
            self.fixed_local_size,
            self.variable_local_size,
            self.variable_global_size,
            self.fixed_global_size,
        )
        if any(size < 0 for size in sizes) or self.history_size <= 0:
            raise ValueError("sizes must be nonnegative and history_size positive")
        if self.token_size == 0:
            raise ValueError("tokens must contain at least one feature")

    @property
    def token_size(self) -> int:
        """Return the raw token width, the encoder's ``input_size``.

        Returns:
            int: ``H * F_vg + F_fg + H * F_vl + F_fl``.
        """
        return (
            self.history_size * (self.variable_global_size + self.variable_local_size)
            + self.fixed_global_size
            + self.fixed_local_size
        )

    def __call__(
        self,
        joint_features: torch.Tensor,
        joint_mask: torch.Tensor | None = None,
        variable_local: torch.Tensor | None = None,
        variable_global: torch.Tensor | None = None,
        fixed_global: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build one token per joint, with zeros at padded joints.

        Args:
            joint_features (torch.Tensor): Fixed local hardware features
                ``[batch, joints, F_fl]``, e.g. ``EmbodimentBatch.joint_features``.
            joint_mask (torch.Tensor | None): Bool ``[batch, joints]`` real joints;
                defaults to all joints.
            variable_local (torch.Tensor | None): ``[batch, H, joints, F_vl]``
                per-joint observations, oldest first. Required if F_vl > 0.
            variable_global (torch.Tensor | None): ``[batch, H, F_vg]`` global
                observations, oldest first. Required if F_vg > 0.
            fixed_global (torch.Tensor | None): ``[batch, F_fg]``. Required if
                F_fg > 0.

        Returns:
            torch.Tensor: Tokens ``[batch, joints, token_size]``.

        Raises:
            ValueError: An input is missing or has the wrong shape.
        """
        if joint_features.ndim != 3 or joint_features.shape[-1] != self.fixed_local_size:
            raise ValueError("joint_features must have shape [batch, joints, F_fl]")
        batch_size, joint_count, _ = joint_features.shape
        history = self.history_size
        parts: list[torch.Tensor] = []
        if self.variable_global_size:
            expected = (batch_size, history, self.variable_global_size)
            if variable_global is None or variable_global.shape != expected:
                raise ValueError(f"variable_global must have shape {expected}")
            flat = variable_global.transpose(1, 2).reshape(batch_size, -1)
            parts.append(flat[:, None].expand(-1, joint_count, -1))
        if self.fixed_global_size:
            expected = (batch_size, self.fixed_global_size)
            if fixed_global is None or fixed_global.shape != expected:
                raise ValueError(f"fixed_global must have shape {expected}")
            parts.append(fixed_global[:, None].expand(-1, joint_count, -1))
        if self.variable_local_size:
            expected = (batch_size, history, joint_count, self.variable_local_size)
            if variable_local is None or variable_local.shape != expected:
                raise ValueError(f"variable_local must have shape {expected}")
            parts.append(
                variable_local.permute(0, 2, 3, 1).reshape(batch_size, joint_count, -1)
            )
        parts.append(joint_features)
        dtype = joint_features.dtype if joint_features.is_floating_point() else None
        tokens = torch.cat([part.to(dtype or torch.float32) for part in parts], dim=-1)
        if joint_mask is not None:
            if joint_mask.shape != (batch_size, joint_count) or joint_mask.dtype != torch.bool:
                raise ValueError("joint_mask must be boolean [batch, joints]")
            tokens = tokens.masked_fill(~joint_mask[..., None], 0)
        return tokens
