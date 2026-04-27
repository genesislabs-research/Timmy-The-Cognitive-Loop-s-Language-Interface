import time
import torch
import torch.nn as nn

QUESTION_LEMMA_SLOTS = {
    "who": 0,
    "what": 1,
    "when": 2,
    "where": 3,
    "why": 4,
    "how": 5,
    "polar_question": 6,
}

I_DONT_KNOW_SLOT = 7
RESERVED_SLOTS = 8

STATUS_UNALLOCATED = 0
STATUS_PROVISIONAL = 1
STATUS_CONFIRMED = 2


class LemmaAcquisitionModule(nn.Module):
    def __init__(
        self,
        n_lemmas: int,
        n_concepts: int,
        n_phonemes: int,
        theta_novelty: float = 0.65,
        theta_production: float = 0.55,
        timeout_seconds: float = 180.0,
    ):
        super().__init__()
        self.n_lemmas = n_lemmas
        self.n_concepts = n_concepts
        self.n_phonemes = n_phonemes
        self.theta_novelty = theta_novelty
        self.theta_production = theta_production
        self.timeout_seconds = timeout_seconds

        self.W_C_to_L = nn.Parameter(
            torch.zeros(n_lemmas, n_concepts), requires_grad=False
        )
        self.W_L_to_P = nn.Parameter(
            torch.zeros(n_phonemes, n_lemmas), requires_grad=False
        )
        self.register_buffer(
            "status",
            torch.zeros(n_lemmas, dtype=torch.long),
        )
        self.register_buffer(
            "allocation_time",
            torch.zeros(n_lemmas, dtype=torch.float64),
        )

        self._initialize_reserved_slots()

    def _initialize_reserved_slots(self) -> None:
        with torch.no_grad():
            for slot_index in range(RESERVED_SLOTS):
                self.status[slot_index] = STATUS_CONFIRMED
                self.allocation_time[slot_index] = 0.0

    def is_novel(self, phonological_code: torch.Tensor) -> bool:
        with torch.no_grad():
            allocated_mask = self.status > STATUS_UNALLOCATED
            if not allocated_mask.any():
                return True
            stored_codes = self.W_L_to_P.t()
            stored_codes_allocated = stored_codes[allocated_mask]
            input_norm = phonological_code / (
                phonological_code.norm() + 1e-8
            )
            stored_norms = stored_codes_allocated / (
                stored_codes_allocated.norm(dim=1, keepdim=True) + 1e-8
            )
            similarities = stored_norms @ input_norm
            max_similarity = similarities.max().item()
            return max_similarity < self.theta_novelty

    def find_free_slot(self) -> int:
        with torch.no_grad():
            for index in range(RESERVED_SLOTS, self.n_lemmas):
                if self.status[index].item() == STATUS_UNALLOCATED:
                    return index
            return -1

    def allocate_row(
        self,
        concept_vector: torch.Tensor,
        phonological_code: torch.Tensor,
    ) -> int:
        with torch.no_grad():
            slot_index = self.find_free_slot()
            if slot_index < 0:
                return -1
            self.W_C_to_L.data[slot_index] = concept_vector
            self.W_L_to_P.data[:, slot_index] = phonological_code
            self.status[slot_index] = STATUS_PROVISIONAL
            self.allocation_time[slot_index] = time.time()
            return slot_index

    def confirm_row(self, slot_index: int) -> None:
        with torch.no_grad():
            if self.status[slot_index].item() == STATUS_PROVISIONAL:
                self.status[slot_index] = STATUS_CONFIRMED

    def decay_unconfirmed(self) -> None:
        with torch.no_grad():
            now = time.time()
            for index in range(RESERVED_SLOTS, self.n_lemmas):
                if self.status[index].item() != STATUS_PROVISIONAL:
                    continue
                age = now - self.allocation_time[index].item()
                if age > self.timeout_seconds:
                    self.status[index] = STATUS_UNALLOCATED
                    self.W_C_to_L.data[index].zero_()
                    self.W_L_to_P.data[:, index].zero_()
                    self.allocation_time[index] = 0.0

    def reinforce_row(
        self,
        slot_index: int,
        concept_vector: torch.Tensor,
        phonological_code: torch.Tensor,
        learning_rate: float = 0.05,
    ) -> bool:
        with torch.no_grad():
            current_status = self.status[slot_index].item()
            if current_status != STATUS_CONFIRMED:
                return False
            self.W_C_to_L.data[slot_index] += learning_rate * (
                concept_vector - self.W_C_to_L.data[slot_index]
            )
            self.W_L_to_P.data[:, slot_index] += learning_rate * (
                phonological_code - self.W_L_to_P.data[:, slot_index]
            )
            return True

    def select_lemma_for_production(
        self,
        concept_vector: torch.Tensor,
    ) -> tuple[int, bool]:
        with torch.no_grad():
            allocated_mask = self.status > STATUS_UNALLOCATED
            if not allocated_mask.any():
                return I_DONT_KNOW_SLOT, False
            scores = self.W_C_to_L @ concept_vector
            scores_allocated = scores.clone()
            scores_allocated[~allocated_mask] = float("-inf")
            best_slot = int(scores_allocated.argmax().item())
            best_score = float(scores_allocated[best_slot].item())
            if best_score < self.theta_production:
                return I_DONT_KNOW_SLOT, False
            polar_q = (
                self.status[best_slot].item() == STATUS_PROVISIONAL
            )
            return best_slot, polar_q


class FrameRecognizer:
    def __init__(
        self,
        n_concepts: int,
        frame_names: list[str] | None = None,
    ):
        if frame_names is None:
            frame_names = [
                "naming",
                "greeting",
                "question_answering",
                "instruction",
                "confirmation",
            ]
        self.n_concepts = n_concepts
        self.frame_names = frame_names
        self.frame_templates: dict[str, torch.Tensor] = {
            name: torch.zeros(n_concepts) for name in frame_names
        }
        self.frame_biases: dict[str, torch.Tensor] = {
            name: torch.zeros(n_concepts) for name in frame_names
        }
        self.current_frame: str | None = None

    def recognize(self, context_vector: torch.Tensor) -> str:
        with torch.no_grad():
            best_score = float("-inf")
            best_frame = self.frame_names[0]
            input_norm = context_vector / (
                context_vector.norm() + 1e-8
            )
            for frame_name, template in self.frame_templates.items():
                template_norm = template / (template.norm() + 1e-8)
                score = float((template_norm @ input_norm).item())
                if score > best_score:
                    best_score = score
                    best_frame = frame_name
            self.current_frame = best_frame
            return best_frame

    def bias_for(self, frame_name: str) -> torch.Tensor:
        return self.frame_biases.get(
            frame_name, torch.zeros(self.n_concepts)
        )


def make_acquisition_optimizer(
    module: nn.Module,
    learning_rate: float,
) -> torch.optim.Optimizer:
    excluded_suffixes = ("W_C_to_L", "W_L_to_P")
    optimizable_params = [
        param
        for name, param in module.named_parameters()
        if not any(name.endswith(suffix) for suffix in excluded_suffixes)
        and param.requires_grad
    ]
    return torch.optim.Adam(optimizable_params, lr=learning_rate)
