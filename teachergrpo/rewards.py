"""Teacher-side reward for TeacherGRPO.

For each teacher rollout the reward combines

* **verified**    - 1 if the extracted answer is correct, else 0
* **alignment**   - negative student/teacher divergence on the rollout, with a
                    curriculum top1 -> top-k -> full KL over a growing share of
                    the hardest tokens
* **length**      - SWLP (Surprisal-Weighted Length Penalty): penalise reasoning
                    steps that both student and teacher find unsurprising
* **answer_pred** - cosine similarity between the student's hidden state on the
                    last tokens and its embedding of the reference answer
"""

import gc
import os
from contextlib import nullcontext
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from teachergrpo.answers import extract_prediction, is_correct


# -----------------------------------------------------------------------------
# Curriculum
# -----------------------------------------------------------------------------

class CurriculumScheduler:
    """Manages curriculum progression across training steps."""

    def __init__(self, total_steps: int, warmup_steps: int = 100):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        """Increment step counter."""
        self.current_step += 1

    def get_progress(self) -> float:
        """Get training progress [0, 1]."""
        if self.current_step < self.warmup_steps:
            return 0.0
        progress = (self.current_step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        return min(1.0, progress)

    def get_alignment_mode(self) -> str:
        """
        Alignment mode for the current progress: top1_kl -> topk_kl -> full_kl.
        """
        progress = self.get_progress()
        if progress < 0.3:
            return "top1_kl"  # CE against the teacher's argmax token
        elif progress < 0.9:
            return "topk_kl"  # KL over the teacher's top-k tokens
        return "full_kl"  # KL over the full vocabulary

    def get_token_percentage(self) -> float:
        """
        Get percentage of tokens to use in KL calculation.
        Gradually increase from 30% to 100%.
        """
        progress = self.get_progress()
        min_pct = 0.3
        max_pct = 1.0
        return min_pct + (max_pct - min_pct) * progress


# -----------------------------------------------------------------------------
# SWLP (Surprisal-Weighted Length Penalty)
# -----------------------------------------------------------------------------


def get_step_segmentation_masks(
    input_ids: torch.Tensor, period_id: int, newline_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Identifies reasoning steps based on delimiters.
    Logic: If delimiters appear consecutively (e.g., ".\n\n"),
    the LAST one is the split point.
    """
    # 1. Base delimiter mask
    is_period = input_ids == period_id
    is_newline = input_ids == newline_id
    is_delimiter = is_period | is_newline  # [B, L]

    # 2. Shift detection to check the next token
    # We want: Current is Delimiter AND Next is NOT Delimiter
    next_is_delimiter = torch.zeros_like(is_delimiter)
    next_is_delimiter[:, :-1] = is_delimiter[:, 1:]  # Shift left

    # 3. Identify Step Ends (The tail of a delimiter chain)
    is_step_end = is_delimiter & (~next_is_delimiter)

    # 4. Generate Step IDs
    # Step ID increases AFTER the step end.
    step_starts = torch.zeros_like(is_step_end)
    step_starts[:, 1:] = is_step_end[:, :-1]  # Shift right to mark start of new step
    step_starts[:, 0] = 1  # Force start at index 0

    step_ids = torch.cumsum(step_starts.long(), dim=-1) - 1  # 0-indexed IDs
    return is_step_end, step_ids


def compute_swlp_reward(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    tokenizer,
    penalty_coefficient: float = 0.01,
    temperature: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Only penalizes steps where BOTH Student and Teacher find the start 'unsurprising'.
    """

    # Dynamic ID detection for robustness across tokenizers
    period_id = tokenizer.encode(".", add_special_tokens=False)[-1]
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]

    # 1. Shift Logits and Labels (Standard Causal LM Logic)
    # logits[t] predicts input_ids[t+1]
    shift_s_logits = student_logits[..., :-1, :].contiguous()
    shift_t_logits = teacher_logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    # 2. Compute Surprisal (NLL)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    s_nll = loss_fct(
        shift_s_logits.view(-1, shift_s_logits.size(-1)), shift_labels.view(-1)
    ).view(shift_labels.shape)
    t_nll = loss_fct(
        shift_t_logits.view(-1, shift_t_logits.size(-1)), shift_labels.view(-1)
    ).view(shift_labels.shape)

    # 3. Compute Unimportance (Probability proxy)
    # High Prob (Low NLL) -> Unimportance ~ 1.0 (Trivial/Redundant)
    # Low Prob (High NLL) -> Unimportance ~ 0.0 (Important/Surprising)
    s_unimp = torch.exp(-s_nll / temperature)
    t_unimp = torch.exp(-t_nll / temperature)

    # Intersection: Both must agree it's trivial to be penalized
    intersection_unimp = s_unimp * t_unimp  # [B, L-1]

    # 4. Step Segmentation
    is_step_end, step_ids = get_step_segmentation_masks(
        shift_labels, period_id, newline_id
    )

    # 5. Identify Anchor (First Token) for each step
    is_first_token = torch.zeros_like(is_step_end)
    is_first_token[:, 0] = True
    is_first_token[:, 1:] = is_step_end[:, :-1]

    # 6. Broadcast First Token Unimportance to the whole step
    batch_size, seq_len = intersection_unimp.shape
    step_weights = torch.zeros_like(intersection_unimp)

    # Vectorized gathering is tricky with variable step counts per batch,
    # using a robust loop over batch (batch size is usually small in GRPO, e.g., 4-16)
    for b in range(batch_size):
        ids = step_ids[b]  # [L-1]
        first_indices = torch.nonzero(is_first_token[b], as_tuple=True)[0]

        # Get values at anchor points
        first_vals = intersection_unimp[b, first_indices]

        # Determine valid range (ids can't exceed number of found steps)
        num_steps = len(first_vals)
        safe_mask = ids < num_steps

        # Fill step_weights: look up the value for the step ID of the current token
        step_weights[b, safe_mask] = first_vals[ids[safe_mask]]

    # 7. Apply Completion Mask (Don't penalize prompt)
    # completion_mask matches input_ids [B, L]. We need [B, L-1]
    valid_mask = completion_mask[..., 1:].float()

    # 8. Calculate Final Penalty per Sample
    # Sum of weighted length
    sample_penalty = torch.sum(step_weights * valid_mask, dim=-1)

    # Reward is negative penalty
    rewards = -penalty_coefficient * sample_penalty

    # Return raw rewards and the mean unimportance for logging
    return rewards, intersection_unimp.mean()


# -----------------------------------------------------------------------------
# Curriculum alignment losses
# -----------------------------------------------------------------------------

def compute_top1_kl_efficient(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Cross-entropy of the student against the teacher's argmax token."""
    teacher_targets = teacher_logits.argmax(dim=-1)
    
    # Clamp to prevent logit explosion
    student_logits = torch.clamp(student_logits, min=-100, max=100)
    
    loss_per_token = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)) / temperature,  # Divide by temperature
        teacher_targets.view(-1),
        reduction='none'
    ).view(student_logits.shape[:-1])
    
    # Also apply nan_to_num for safety
    loss_per_token = torch.nan_to_num(loss_per_token, nan=0.0)
    
    return loss_per_token


def compute_topk_kl_efficient(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    k: int = 100,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Memory-efficient top-k KL using log_softmax for stability."""
    batch_size, seq_len, vocab_size = student_logits.shape
    device = student_logits.device
    loss_per_token = torch.zeros(batch_size, seq_len, device=device)
    
    for b in range(batch_size):
        # 1. Get Teacher's Top-K
        topk_vals, topk_indices = teacher_logits[b].topk(k, dim=-1)
        
        # 2. Align Student logits to Teacher's top-k indices
        student_topk = torch.gather(student_logits[b], -1, topk_indices)
        
        # 3. Use log_softmax for numerical stability
        # Teacher
        t_log_probs = F.log_softmax(topk_vals / temperature, dim=-1)
        t_probs = torch.exp(t_log_probs)
        
        # Student
        s_log_probs = F.log_softmax(student_topk / temperature, dim=-1)
        
        # 4. Compute KL: p * (log_p - log_q)
        # t_log_probs and s_log_probs are already in log-space
        kl = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1)
        
        loss_per_token[b] = kl
        
    return loss_per_token

def compute_full_kl_efficient(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Memory-efficient full KL using log_softmax."""
    batch_size, seq_len, vocab_size = student_logits.shape
    device = student_logits.device
    loss_per_token = torch.zeros(batch_size, seq_len, device=device)
    
    chunk_size = 128
    for start_idx in range(0, seq_len, chunk_size):
        end_idx = min(start_idx + chunk_size, seq_len)
        
        s_chunk = student_logits[:, start_idx:end_idx, :]
        t_chunk = teacher_logits[:, start_idx:end_idx, :]
        
        # Key Change: Use log_softmax
        t_log_probs = F.log_softmax(t_chunk / temperature, dim=-1)
        s_log_probs = F.log_softmax(s_chunk / temperature, dim=-1)
        t_probs = torch.exp(t_log_probs)
        
        kl_chunk = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1)
        loss_per_token[:, start_idx:end_idx] = kl_chunk
    
    return loss_per_token


def select_topk_tokens_by_kl(
    kl_per_token: torch.Tensor, 
    mask: torch.Tensor, 
    top_percentage: float
) -> torch.Tensor:
    """Memory-efficient top-k token selection."""
    batch_size, seq_len = kl_per_token.shape
    new_mask = torch.zeros_like(mask)
    
    valid_tokens = mask.sum(dim=1)
    k_per_seq = (valid_tokens * top_percentage).long().clamp(min=1)
    
    for i in range(batch_size):
        k = k_per_seq[i].item()
        if k > 0:
            masked_kl = kl_per_token[i].clone()
            masked_kl[mask[i] == 0] = -float('inf')
            topk_indices = masked_kl.topk(k, largest=True).indices
            new_mask[i, topk_indices] = 1
            del masked_kl, topk_indices
    
    return new_mask


# -----------------------------------------------------------------------------
# Memory-efficient alignment loss
# -----------------------------------------------------------------------------

def compute_alignment_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
    temperature: float = 1.0,
    k: int = 100,
) -> torch.Tensor:
    """Per-token alignment loss for the given curriculum mode, masked."""
    if mode == "ce" or mode == "top1_kl":
        loss_per_token = compute_top1_kl_efficient(
            student_logits, teacher_logits, temperature=temperature
        )
    elif mode == "topk_kl":
        loss_per_token = compute_topk_kl_efficient(
            student_logits, teacher_logits, k=k, temperature=temperature
        )
    else:  # full_kl
        loss_per_token = compute_full_kl_efficient(
            student_logits, teacher_logits, temperature=temperature
        )
    
    loss_per_token = loss_per_token * mask
    if torch.isnan(loss_per_token).any() or torch.isinf(loss_per_token).any():
        loss_per_token = torch.nan_to_num(loss_per_token, nan=0.0, posinf=0.0, neginf=0.0)
    return loss_per_token


# -----------------------------------------------------------------------------
# Main Reward Computation with Curriculum
# -----------------------------------------------------------------------------


class CurriculumRewardFunction:
    """
    Comprehensive reward function with curriculum learning and SWLP logic.
    """

    def __init__(
        self,
        student_model,
        teacher_model,
        tokenizer,
        curriculum_scheduler: CurriculumScheduler,
        eval_type: str = "math",
        device: str = "cuda",
        # Reward weights
        w_verified: float = 1.0,
        w_alignment: float = 0.5,
        w_length: float = 0.2,  # weight of the SWLP length reward
        w_answer_pred: float = 0.3,
        # SWLP hyperparameters
        swlp_beta: float = 0.02,  # penalty coefficient
        swlp_temperature: float = 0.5,  # sensitivity to surprisal
        n_answer_tokens: int = 20,
        max_length: int = 2048,
    ):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.scheduler = curriculum_scheduler
        self.eval_type = eval_type
        self.device = device
        self.max_length = max_length

        # Weights
        self.w_verified = w_verified
        self.w_alignment = w_alignment
        self.w_length = w_length
        self.w_answer_pred = w_answer_pred

        # SWLP Params
        self.swlp_beta = swlp_beta
        self.swlp_temperature = swlp_temperature

        self.n_answer_tokens = n_answer_tokens

        self.__name__ = "curriculum_reward"
        self.last_metrics = {}

    def _stash_metrics(self, metrics: Dict[str, float]):
        self.last_metrics = {k: float(v) for k, v in metrics.items()}

    def _clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()

    def __call__(
        self,
        prompts: List[List[Dict]],
        completions: List[List[Dict]],
        reference: List[str] = None,
        **kwargs
    ) -> List[float]:
        """
        Memory-optimized reward computation with micro-batching.
        """
        
        # 1. Initial cleanup
        self._clear_memory()
        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        
        if reference is None:
            reference = kwargs.get('answer', kwargs.get('answers', []))
            if not reference:
                reference = [""] * len(prompts)
        
        batch_size = len(prompts)
        
        # Extract text
        prompt_texts = []
        completion_texts = []
        for p, c in zip(prompts, completions):
            if isinstance(p, list): 
                p_content = p[0].get('content', '') if isinstance(p[0], dict) else str(p[0])
            else: 
                p_content = str(p)
            if isinstance(c, list): 
                c_content = c[0].get('content', '') if isinstance(c[0], dict) else str(c[0])
            else: 
                c_content = str(c)
            prompt_texts.append(p_content)
            completion_texts.append(c_content)
        
        full_texts = [p + c for p, c in zip(prompt_texts, completion_texts)]
        
        # Get curriculum settings
        alignment_mode = self.scheduler.get_alignment_mode()
        token_percentage = self.scheduler.get_token_percentage()

        # =====================================================================
        # Pre-compute verified rewards (CPU only, cheap)
        # =====================================================================
        verified_rewards = torch.tensor(
            [
                1.0 if is_correct(extract_prediction(c, self.eval_type), ref, self.eval_type) else 0.0
                for c, ref in zip(completion_texts, reference)
            ],
            device=self.device,
        )

        # =====================================================================
        # Micro-batching strategy
        # =====================================================================
        micro_batch_size = 4  # Reduce to 1 if GPU memory is limited
        all_alignment_rewards = []
        all_swlp_rewards = []
        all_answer_pred_rewards = []
        mean_redundancies = []
        
        is_gemma = "gemma" in self.student_model.config._name_or_path
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if is_gemma else nullcontext()
        
        with torch.no_grad(), autocast_ctx:
            # Pre-compute reference embeddings (shared across micro-batches)
            ref_inputs = self.tokenizer(
                reference, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=64
            ).to(self.device)
            
            ref_outputs = self.student_model(
                input_ids=ref_inputs.input_ids, 
                output_hidden_states=True
            )
            ref_embeddings = ref_outputs.hidden_states[-1].mean(dim=1).clone()
            del ref_outputs, ref_inputs
            self._clear_memory()
            
            # Process in micro-batches
            for i in range(0, batch_size, micro_batch_size):
                end_idx = min(i + micro_batch_size, batch_size)
                micro_texts = full_texts[i:end_idx]
                micro_prompts = prompt_texts[i:end_idx]
                
                # Tokenize micro-batch
                inputs = self.tokenizer(
                    micro_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)
                
                # Generate completion mask
                completion_mask = inputs.attention_mask.clone().float()
                for j, p_text in enumerate(micro_prompts):
                    prompt_tokens = self.tokenizer(
                        p_text, 
                        add_special_tokens=True, 
                        truncation=False
                    ).input_ids
                    if isinstance(prompt_tokens[0], list): 
                        prompt_tokens = prompt_tokens[0]
                    p_len = len(prompt_tokens)
                    mask_len = min(p_len, completion_mask.shape[1])
                    completion_mask[j, :mask_len] = 0.0
                
                # =====================================================================
                # Sequential model calls with immediate cleanup
                # =====================================================================
                
                # Student forward
                student_outputs = self.student_model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True
                )
                student_logits = student_outputs.logits.clone()
                student_hidden = student_outputs.hidden_states[-1].clone()
                del student_outputs
                torch.cuda.empty_cache()
                
                # Teacher forward
                teacher_outputs = self.teacher_model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=False
                )
                teacher_logits = teacher_outputs.logits.clone()
                del teacher_outputs
                torch.cuda.empty_cache()
                
                # =====================================================================
                # Alignment Reward (memory-efficient)
                # =====================================================================
                alignment_loss = compute_alignment_loss(
                    student_logits, 
                    teacher_logits, 
                    completion_mask,
                    mode=alignment_mode, 
                    temperature=1.0,
                    k=100,
                )
                
                token_mask = select_topk_tokens_by_kl(
                    alignment_loss, 
                    completion_mask, 
                    top_percentage=token_percentage
                )
                
                masked_alignment = (alignment_loss * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp(min=1)
                alignment_rewards_micro = -masked_alignment
                all_alignment_rewards.append(alignment_rewards_micro.cpu())
                
                del alignment_loss, token_mask, masked_alignment
                torch.cuda.empty_cache()
                
                # =====================================================================
                # SWLP Length Reward
                # =====================================================================
                swlp_rewards_micro, mean_redundancy = compute_swlp_reward(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    input_ids=inputs.input_ids,
                    completion_mask=completion_mask,
                    tokenizer=self.tokenizer,
                    penalty_coefficient=self.swlp_beta,
                    temperature=self.swlp_temperature
                )
                all_swlp_rewards.append(swlp_rewards_micro.cpu())
                mean_redundancies.append(mean_redundancy.cpu())
                
                # Delete logits immediately after use
                del teacher_logits, student_logits
                torch.cuda.empty_cache()
                
                # =====================================================================
                # Answer Prediction Reward
                # =====================================================================
                answer_pred_rewards_micro = torch.zeros(end_idx - i, device=self.device)
                for j in range(end_idx - i):
                    full_len = inputs.attention_mask[j].sum().item()
                    if full_len > self.n_answer_tokens:
                        start_idx = int(full_len - self.n_answer_tokens)
                        end_idx_slice = int(full_len)
                        curr_embed = student_hidden[j, start_idx:end_idx_slice, :].mean(dim=0)
                        
                        similarity = F.cosine_similarity(
                            curr_embed.unsqueeze(0), 
                            ref_embeddings[i + j].unsqueeze(0), 
                            dim=-1
                        )
                        answer_pred_rewards_micro[j] = similarity.item()
                    else:
                        answer_pred_rewards_micro[j] = 0.5
                
                all_answer_pred_rewards.append(answer_pred_rewards_micro.cpu())
                
                # Cleanup micro-batch
                del student_hidden, inputs, completion_mask
                torch.cuda.empty_cache()
            
            # Final cleanup
            del ref_embeddings
            self._clear_memory()
        
        # =====================================================================
        # Combine results on CPU then move to GPU
        # =====================================================================
        alignment_rewards = torch.cat(all_alignment_rewards).to(self.device)
        swlp_rewards = torch.cat(all_swlp_rewards).to(self.device)
        answer_pred_rewards = torch.cat(all_answer_pred_rewards).to(self.device)
        mean_redundancy = torch.stack(mean_redundancies).mean()
        
        # Normalize and combine
        alignment_norm = alignment_rewards
        length_norm = swlp_rewards
        answer_norm = (answer_pred_rewards + 1) / 2

        complex_reward = (
            (alignment_norm * self.w_alignment)
            + (length_norm * self.w_length)
            + (answer_norm * self.w_answer_pred)
        )

        final_reward = verified_rewards * self.w_verified + complex_reward

        # Metrics
        metrics = {
            "reward/final_mean": final_reward.mean().item(),
            "reward/verified_mean": verified_rewards.mean().item(),
            "reward/alignment_mean": alignment_norm.mean().item(),
            "reward/swlp_length_penalty_mean": length_norm.mean().item(),
            "reward/redundancy_score": mean_redundancy.item(),
            "reward/answer_pred_mean": answer_norm.mean().item(),
            "curriculum/progress": self.scheduler.get_progress(),
        }
        self._stash_metrics(metrics)

        return final_reward.cpu().tolist()
