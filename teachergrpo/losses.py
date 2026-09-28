"""Token-level distillation losses between student and teacher logits.

All divergences take logits of shape ``[..., V]`` and an optional ``mask`` of
shape ``[...]``. When vocabularies differ in size, the shared prefix is used.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def compute_sft_loss(logits: torch.Tensor, labels: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Next-token cross-entropy; positions labelled -100 are ignored."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction=reduction,
    )


def _log_probs(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    vocab = min(student_logits.size(-1), teacher_logits.size(-1))
    student_log_probs = torch.log_softmax(student_logits[..., :vocab] / temperature, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits[..., :vocab] / temperature, dim=-1)
    return student_log_probs, teacher_log_probs


def _reduce(per_position: torch.Tensor, mask: Optional[torch.Tensor], reduction: str) -> torch.Tensor:
    if mask is not None:
        per_position = per_position * mask
    if reduction == "none":
        return per_position
    if reduction == "sum":
        return per_position.sum()
    if reduction == "mean":
        if mask is not None:
            return per_position.sum() / mask.sum().clamp(min=1)
        return per_position.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


def forward_kl_div_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    temperature: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """KL(teacher || student), scaled by temperature^2."""
    s_log_probs, t_log_probs = _log_probs(student_logits, teacher_logits, temperature)
    kl = F.kl_div(s_log_probs, t_log_probs, reduction="none", log_target=True).sum(dim=-1)
    return _reduce(kl * temperature ** 2, mask, reduction)


def reverse_kl_div_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    temperature: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """KL(student || teacher), scaled by temperature^2."""
    s_log_probs, t_log_probs = _log_probs(student_logits, teacher_logits, temperature)
    kl = F.kl_div(t_log_probs, s_log_probs, reduction="none", log_target=True).sum(dim=-1)
    return _reduce(kl * temperature ** 2, mask, reduction)


def generalized_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    beta: float = 0.5,
    temperature: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Generalised Jensen-Shannon divergence with mixture M = beta*P_T + (1-beta)*P_S."""
    s_log_probs, t_log_probs = _log_probs(student_logits, teacher_logits, temperature)
    mixture = torch.logsumexp(
        torch.stack([math.log(beta) + t_log_probs, math.log(1.0 - beta) + s_log_probs], dim=0), dim=0
    )
    kl_teacher = F.kl_div(mixture, t_log_probs, reduction="none", log_target=True)
    kl_student = F.kl_div(mixture, s_log_probs, reduction="none", log_target=True)
    jsd = (beta * kl_teacher + (1.0 - beta) * kl_student).sum(dim=-1)
    return _reduce(jsd, mask, reduction)
