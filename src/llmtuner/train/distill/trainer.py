from collections import defaultdict
from contextlib import nullcontext

import torch
from transformers import BatchEncoding, Trainer
from transformers import Seq2SeqTrainer
from trl.trainer.utils import disable_dropout_in_model

from ...extras.constants import IGNORE_INDEX
from ..utils import create_custom_optimzer, create_custom_scheduler

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ...extras.logging import get_logger

from torch import nn

from copy import deepcopy
from trl.models import PreTrainedModelWrapper
import deepspeed

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)

class DistillationLoss(nn.Module):
    def __init__(self , alpha=0.5):
        """
        Initializes the DistillationLoss module.
        :param temperature: Temperature for softmax scaling, making the teacher's distribution softer.
        :param alpha: Weighting factor for balancing the soft and hard loss components.
        """
         
        super().__init__()
        self.alpha = alpha
        self.soft_loss = nn.KLDivLoss(reduction='batchmean')
        self.temperature = 1 #Adjust this later

    def forward(self, model_logits, ref_logits, labels, hard_loss_value):
        """
        Forward pass for the distillation loss computation.
        :param student_logits: Logits from the student model.
        :param teacher_logits: Logits from the teacher model.
        :param labels: Ground truth labels.
        :return: Combined distillation loss.
        """
        # ##(forward KL divergence loss)
        # # Calculate soft logits with temperature scaling
        # soft_logits = nn.functional.log_softmax(model_logits / self.temperature, dim=1)
        # with torch.no_grad():
        #     soft_targets = nn.functional.softmax(ref_logits / self.temperature, dim=1)
        # # Calculate the soft loss 
        # soft_loss = self.soft_loss(soft_logits, soft_targets) * (self.alpha * self.temperature ** 2)

        ##(reverse KL divergence loss)
        # Calculate soft logits with temperature scaling
        soft_logits = nn.functional.log_softmax(ref_logits / self.temperature, dim=1)
        with torch.no_grad():
            soft_targets = nn.functional.softmax(model_logits / self.temperature, dim=1)
        # Calculate the soft loss 
        soft_loss = self.soft_loss(soft_logits, soft_targets) * (self.alpha * self.temperature ** 2)

        hard_loss = (1 - self.alpha) * hard_loss_value
        
        # Return the total loss as a weighted sum of the soft and hard losses
        return soft_loss + hard_loss

from transformers.modeling_utils import unwrap_model

class CustomDistillTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.finetuning_args = finetuning_args
        self.ref_model = ref_model

        self.reference_free = False
        self.use_dpo_data_collator = False  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs) #If an optimizer is specified in the deepspeed_config this won't work (largest numel = max([sum([p.ds_numel for p in psg]) for psg in self.fp16 partitioned groups]))
        model.eval()
        return model

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"].detach().clone() if "labels" in inputs else None  # backup labels
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )

        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: torch.Tensor, tgt_tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):
                preds[i] = np.concatenate(
                    (preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1
                )  # move pad token to last

        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for label, pred in zip(decoded_labels, decoded_preds):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))

    def compute_loss(self, model, inputs, return_outputs=False):

        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        model_outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = model_outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                hard_loss = self.label_smoother(model_outputs, labels, shift_labels=True)
            else:
                hard_loss = self.label_smoother(model_outputs, labels)
        else:
            if isinstance(model_outputs, dict) and "loss" not in model_outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(model_outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            hard_loss = model_outputs["loss"] if isinstance(model_outputs, dict) else model_outputs[0]

        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        model_outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = model_outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                hard_loss = self.label_smoother(model_outputs, labels, shift_labels=True)
            else:
                hard_loss = self.label_smoother(model_outputs, labels)
        else:
            if isinstance(model_outputs, dict) and "loss" not in model_outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(model_outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            hard_loss = model_outputs["loss"] if isinstance(model_outputs, dict) else model_outputs[0]

        model_logits = model_outputs["logits"] if isinstance(model_outputs, dict) else model_outputs[1]
        model_logits = model_outputs["logits"] if isinstance(model_outputs, dict) else model_outputs[1]

        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs)
            ref_logits = ref_outputs["logits"] if isinstance(ref_outputs, dict) else ref_outputs[1]
            ref_outputs = self.ref_model(**inputs)
            ref_logits = ref_outputs["logits"] if isinstance(ref_outputs, dict) else ref_outputs[1]

        loss_fn = DistillationLoss(self.finetuning_args.distill_teacher_mixin) 
        loss = loss_fn(model_logits, ref_logits, labels, hard_loss_value = hard_loss)
        loss = loss_fn(model_logits, ref_logits, labels, hard_loss_value = hard_loss)

        return (loss, model_outputs) if return_outputs else loss