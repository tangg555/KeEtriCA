"""
@Desc:
@Reference:
- BART shift_tokens_right
https://huggingface.co/docs/transformers/v4.17.0/en/model_doc/bart#bart
- Label Smoothing
https://paperswithcode.com/method/label-smoothing
- bart models from huggingface
e.g. https://huggingface.co/facebook/bart-base
@Notes:
- BART shift_tokens_right
Bart uses the eos_token_id as the starting token for decoder_input_ids generation.
If past_key_values is used, optionally only the last decoder_input_ids have to be input (see past_key_values).
For translation and summarization training, decoder_input_ids should be provided. If no decoder_input_ids is provided,
the model will create this tensor by shifting the input_ids to the right for denoising pre-training following the paper.
- label-smoothing
During finetuning we use a label smoothed cross entropy loss (Pereyra et al., 2017), with the smoothing parameter
set to 0.1.
- model generate:
in generation_utils.py e.g.BartForConditionalGeneration().generate -> def generate in generation_utils.py
- torch.nn.CrossEntropyLoss
    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: If containing class indices, shape :math:`(N)` where each value is
          :math:`0 \leq \text{targets}[i] \leq C-1`, or :math:`(N, d_1, d_2, ..., d_K)` with
          :math:`K \geq 1` in the case of K-dimensional loss. If containing class probabilities,
          same shape as the input.
"""

import logging
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from transformers.models.bart import modeling_bart
from transformers.models.bart.modeling_bart import BartForConditionalGeneration, BartConfig
from transformers import BartTokenizer

from src.utils.gen_utils import ids_to_clean_string, top_p_logits
from src.modules.event_trigger.datasets import (
    EventLineDataset,
    LeadingContextDataset,
    LeadingPlusEventDataset,
    LeadingToEventsDataset
)
from src.utils.event_trigger import model_utils
from src.utils import nlg_eval_utils
from src.utils.file_utils import save_json, pickle_save
from src.models.lightning_base import BaseTransformer

logger = logging.getLogger(__name__)


class EventBart(BaseTransformer):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

        self._custom_init()

        # Whether changing embeddings
        if self.hparams.freeze_embeds:
            model_utils.freeze_embeds(self.model)
        if self.hparams.freeze_encoder:
            model_utils.freeze_params(self.model.get_encoder())
            model_utils.assert_all_frozen(self.model.get_encoder())

        self.step_count = 0
        self.current_val_metrics = {}
        self.metrics_save_path = Path(self.experiment_output_dir) / "metrics.json"
        self.metrics: dict = defaultdict(list)
        self.model_type = self.config.model_type
        self.decoder_start_token_id = self.model.config.decoder_start_token_id  # default to config
        self.already_saved_batch = False  # flag of saving readable batch
        self.eval_beams = self.model.config.num_beams if self.hparams.eval_beams is None else self.hparams.eval_beams
        self.val_metric = "loss" if self.hparams.val_metric is None else self.hparams.val_metric
        self.save_readable_batch = True  # for debug
        self.metric_names_update_flag = True

        # predicted
        self.use_top_p = False
        self.top_p = 0.9
        self.store_test_output = True
        self.test_output = None
        self.remain_sp_tokens = self.hparams.remain_sp_tokens
        if self.remain_sp_tokens:
            print("remain special tokens in target and pred text (e.g. [EVENT_s])")


    def _custom_init(self):
        # load pretrained settings from bart
        # config
        self.config: BartConfig = BartConfig.from_pretrained(self.hparams.model_name_or_path)
        # tokenizer
        self.tokenizer: BartTokenizer = BartTokenizer.from_pretrained(self.hparams.model_name_or_path)
        # model
        self.model = self._load_model(self.hparams.model_name_or_path, BartForConditionalGeneration, self.config)
        self._set_up(config=self.config,
                     tokenizer=self.tokenizer,
                     model=self.model)
        self.dataset_class = EventLineDataset

        self.train_event_infix = "_event"
        self.test_event_infix = self.hparams.test_event_infix if self.hparams.test_event_infix else "_predicted_event"
        self.eval_event_infix = self.test_event_infix

    def save_readable_batch_fn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        """A debugging utility"""
        readable_batch = {
            k: self.tokenizer.batch_decode(v.tolist()) if k not in ["ids"] else v for k, v in batch.items()
        }
        save_json(readable_batch, Path(self.experiment_output_dir) / "text_batch.json")
        save_json({k: v.tolist() if k not in ["ids"] else v for k, v in batch.items()},
                  Path(self.experiment_output_dir) / "tok_batch.json")

        self.already_saved_batch = True
        return readable_batch

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def gather_nd(self, x, indices):
        newshape = indices.shape[:-1] + x.shape[indices.shape[-1]:]
        indices = indices.view(-1, indices.shape[-1]).tolist()
        out = torch.cat([x.__getitem__(tuple(i)) for i in indices]).reshape(newshape)
        return out

    def _step(self, batch: dict):
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        tgt_ids = batch["labels"]

        decoder_input_ids = modeling_bart.shift_tokens_right(tgt_ids,
                                                             self.pad_token_id,
                                                             self.decoder_start_token_id)
        if self.save_readable_batch and not self.already_saved_batch:
            # This would be slightly better if it only happened on rank zero
            batch["decoder_input_ids"] = decoder_input_ids
            self.save_readable_batch_fn(batch)
        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False,
                       output_attentions=True, output_hidden_states=True)

        lm_logits = outputs["logits"]

        if self.hparams.label_smoothing == 0:
            assert lm_logits.shape[-1] == self.vocab_size
            # lm_ligits: [batch, seq, vocab] tgt_ids: [batch, seq]
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction="none")
            losses_ = self.loss_fn(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
            loss = torch.mean(losses_)
        else:
            lprobs = torch.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = model_utils.label_smoothed_nll_loss(
                lprobs, tgt_ids, self.hparams.label_smoothing, ignore_index=self.pad_token_id
            )
        lm_loss = loss
        return lm_loss

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx) -> Dict:
        loss = self._step(batch)
        logs = {"loss": loss.item()}
        # metrics logged can be access by trainer.callback_metrics
        self.log_dict(self.current_val_metrics)
        # tokens per batch
        logs["tokens_per_batch"] = batch["input_ids"].ne(self.pad_token_id).sum() + batch["labels"].ne(
            self.pad_token_id).sum()
        logs["batch_size"] = batch["input_ids"].shape[0]
        logs["source_pad_tokens_num"] = batch["input_ids"].eq(self.pad_token_id).sum()
        logs["source_pad_tokens_ratio"] = batch["input_ids"].eq(self.pad_token_id).float().mean()
        return {"loss": loss, "log": logs}

    @torch.no_grad()
    def sample_sequence(self, batch, use_top_p=False, top_p=0.9):
        batch_size = len(batch["ids"])
        decoder_input_ids = torch.tensor([self.tokenizer.eos_token_id for _
                                          in range(batch_size)])[:, None].to(self.device)
        for _ in range(self.hparams.max_target_length):
            outputs = self(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"],
                           decoder_input_ids=decoder_input_ids,
                           use_cache=False, return_dict=True)
            logits = outputs["logits"]
            logits = logits[:, -1, :]
            if use_top_p:
                logits = top_p_logits(logits, p=top_p, device=self.device)
                probs = torch.softmax(logits, dim=-1)
                pred = torch.multinomial(probs, 1)
            else:
                probs = torch.softmax(logits, dim=-1)
                pred = torch.topk(input=probs, k=1).indices
            decoder_input_ids = torch.cat([decoder_input_ids, pred], 1)
            # early stop
            if pred[:, 0].eq(self.tokenizer.eos_token_id).sum() == pred.shape[0]:
                break
        generated_ids = decoder_input_ids
        return generated_ids

    def gen_ids_to_clean_text(self, generated_ids: List[int]):
        gen_list = []
        for output in generated_ids:
            gen_list.append(ids_to_clean_string(output, self.tokenizer, remain_sp_tokens=self.remain_sp_tokens))
        return gen_list

    @torch.no_grad()
    def _generative_step(self, batch: dict, fast_generate=False) -> dict:
        tik = datetime.now()
        if fast_generate:
            generated_ids = self.model.model_generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=True,
                decoder_start_token_id=self.decoder_start_token_id,
                num_beams=self.eval_beams,
                max_length=self.eval_max_length,
            )
        else:
            generated_ids = self.sample_sequence(batch, use_top_p=self.use_top_p, top_p=self.top_p)
        tok = datetime.now()
        batch_gen_time = tok - tik
        preds: List[str] = self.gen_ids_to_clean_text(generated_ids)
        targets: List[str] = self.gen_ids_to_clean_text(batch["labels"])
        loss = self._step(batch)

        base_metrics = {"loss": loss.item()}
        rouge_metrics: Dict = nlg_eval_utils.calculate_rouge(pred_lines=preds, tgt_lines=targets)
        base_metrics.update(**rouge_metrics)
        bleu_metrics: Dict = nlg_eval_utils.calculate_bleu(ref_lines=[self.tokenizer.tokenize(l) for l in targets],
                                                           gen_lines=[self.tokenizer.tokenize(l) for l in preds])
        base_metrics.update(**bleu_metrics)
        summ_len = np.mean(list(map(len, generated_ids)))

        # update metric_names
        self.update_metric_names(base_metrics, update_flag=self.metric_names_update_flag)
        self.metric_names_update_flag = False
        base_metrics.update(batch_gen_time=batch_gen_time, gen_len=summ_len,
                            preds=preds, targets=targets)
        return base_metrics

    def validation_step(self, batch, batch_idx) -> Dict:
        try:
            return self._generative_step(batch, fast_generate=self.hparams.fast_generate)
        except Exception as e:
            error_file_path = self.experiment_output_dir / "eval_error_batch.pkl"
            pickle_save(batch, error_file_path)
            print(f"error occur: {e}\nthe batch: {batch}\noutput error batch to {error_file_path}")
            raise e

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        generative_metrics = {
            name: np.array([x[name] for x in outputs]).mean() for name in self.metric_names
        }
        metric_val = (
            torch.tensor(generative_metrics[self.val_metric])
        )
        val_metrics = {f"{prefix}_{k}": x for k, x in generative_metrics.items()}
        val_metrics["step_count"] = float(self.step_count)
        self.current_val_metrics = val_metrics
        self.metrics[prefix].append(val_metrics)  # callback writes this to self.metrics_save_path.
        print(f"Evaluation result: {val_metrics}")
        preds = model_utils.flatten_list([x["preds"] for x in outputs])
        tgts = model_utils.flatten_list([x["targets"] for x in outputs])
        self.log_dict(self.current_val_metrics)
        return {
            "log": val_metrics,
            "preds": preds,
            "tgts": tgts,
            f"{prefix}_loss": generative_metrics["loss"],
            f"{prefix}_{self.val_metric}": metric_val,
        }

    def test_step(self, batch, batch_idx):
        try:
            return self._generative_step(batch, fast_generate=self.hparams.fast_generate)
        except Exception as e:
            error_file_path = self.experiment_output_dir / "test_error_batch.pt"
            torch.save(batch, error_file_path)
            print(f"error occur: {e}\nthe batch: {batch}\noutput error batch to {error_file_path}")
            raise e

    def test_epoch_end(self, outputs):
        test_output = self.validation_epoch_end(outputs, prefix="test")
        if self.store_test_output:
            self.test_output = test_output
        return test_output

    def get_dataset(self, src_file_prefix: str, tgt_file_prefix: str) -> EventLineDataset:
        event_infix = ""
        if "train" in src_file_prefix:
            event_infix = self.train_event_infix
        elif "val" in src_file_prefix:
            event_infix = self.eval_event_infix
        elif "test" in src_file_prefix:
            event_infix = self.test_event_infix
        else:
            NotImplementedError()

        dataset = self.dataset_class(
            self.tokenizer,
            src_file_prefix=f"{src_file_prefix}"
                              f"{event_infix}",
            tgt_file_prefix=tgt_file_prefix,
            max_target_length=self.hparams.max_target_length,
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
        )
        self.model.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
        return dataset

    def get_dataloader(self, src_file_prefix: str, tgt_file_prefix: str,
                       batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(src_file_prefix, tgt_file_prefix)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        train_shuffle = True if self.hparams.overfit_batches == 0.0 else False
        if not train_shuffle:
            print(f"train_shuffle: {train_shuffle} overfit_batches: {self.hparams.overfit_batches}")
        return self.get_dataloader("train", "train", batch_size=self.hparams.train_batch_size,
                                   shuffle=train_shuffle)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", "val", batch_size=self.hparams.eval_batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", "test", batch_size=self.hparams.eval_batch_size, shuffle=False)


class LeadingContextBart(EventBart):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

    def _custom_init(self):
        super()._custom_init()
        self.dataset_class = LeadingContextDataset

    def get_dataset(self, src_file_prefix: str, tgt_file_prefix: str) -> EventLineDataset:
        dataset = self.dataset_class(
            self.tokenizer,
            src_file_prefix=f"{src_file_prefix}",
            tgt_file_prefix=tgt_file_prefix,
            max_target_length=self.hparams.max_target_length,
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
        )
        self.model.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
        return dataset

class LeadingToEventsBart(LeadingContextBart):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

    def _custom_init(self):
        super()._custom_init()
        self.dataset_class = LeadingToEventsDataset

    def ids_to_clean_string(self, token_list, tokenizer):
        # leading to events is special, because we need special tokens of EventGraph
        real_s = 0
        for index_, token_ in enumerate(token_list):
            if token_ not in [tokenizer.bos_token_id, tokenizer.eos_token_id]:
                real_s = index_
                break
        token_list = token_list[real_s:]
        string = tokenizer.decode(token_list, skip_special_tokens=False)
        string = string[:string.find(tokenizer.eos_token)].strip()
        for one in [tokenizer.bos_token, tokenizer.eos_token]:
            string = string.replace(one, " ")
        string = " ".join([one for one in string.split(" ")])
        return string

    def train_dataloader(self) -> DataLoader:
        train_shuffle = True if self.hparams.overfit_batches == 0.0 else False
        if not train_shuffle:
            print(f"train_shuffle: {train_shuffle} overfit_batches: {self.hparams.overfit_batches}")
        return self.get_dataloader("train", "train_event", batch_size=self.hparams.train_batch_size,
                                   shuffle=train_shuffle)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", "val_event", batch_size=self.hparams.eval_batch_size,
                                   shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", "test_event", batch_size=self.hparams.eval_batch_size,
                                   shuffle=False)


class LeadingPlusEventBart(EventBart):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

    def _custom_init(self):
        super()._custom_init()
        self.dataset_class = LeadingPlusEventDataset
        self.train_event_infix = "_event"
        self.test_event_infix = self.hparams.test_event_infix if self.hparams.test_event_infix else "_predicted_event"
        self.eval_event_infix = self.test_event_infix

    def get_dataset(self, src_file_prefix: str, tgt_file_prefix: str) -> LeadingPlusEventDataset:
        event_infix = ""
        if "train" in src_file_prefix:
            event_infix = self.train_event_infix
        elif "val" in src_file_prefix:
            event_infix = self.eval_event_infix
        elif "test" in src_file_prefix:
            event_infix = self.test_event_infix
        else:
            NotImplementedError()
        dataset = self.dataset_class(
            self.tokenizer,
            src_file_prefix=src_file_prefix,
            event_file_prefix=f"{src_file_prefix}"
                              f"{event_infix}",
            tgt_file_prefix=tgt_file_prefix,
            max_target_length=self.hparams.max_target_length,
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
        )
        self.model.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
        return dataset

    def train_dataloader(self) -> DataLoader:
        train_shuffle = True if self.hparams.overfit_batches == 0.0 else False
        if not train_shuffle:
            print(f"train_shuffle: {train_shuffle} overfit_batches: {self.hparams.overfit_batches}")
        return self.get_dataloader("train", "train", batch_size=self.hparams.train_batch_size,
                                   shuffle=train_shuffle)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", "val", batch_size=self.hparams.eval_batch_size,
                                   shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", "test", batch_size=self.hparams.eval_batch_size,
                                   shuffle=False)
