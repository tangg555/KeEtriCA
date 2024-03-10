#!/usr/bin/env python
"""
@Desc:
@Reference:
- MBART
The MBart model was presented in Multilingual Denoising Pre-training for Neural Machine Translation.
https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/mbart#transformers.MBartTokenizer
- SBERT
https://zhuanlan.zhihu.com/p/383138444
《Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks》对预训练的BERT进行修改：使用孪生(Siamese)
和三级(triplet)网络结构来获得语义上有意义的句子embedding，以此获得定长的sentence embedding，使用余弦相似度或Manhatten/Euclidean距离
等进行比较找到语义相似的句子。
@Notes:
"""


import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from transformers import MBartTokenizer, T5ForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right

from src.modules.thu_coai_hint.datasets import (
    Seq2SeqDataset,
    Seq2SeqSbertscoreOrderDisDataset,
)

from src.utils.thu_coai_hint.model_utils import (
    assert_all_frozen,
    calculate_bleu,
    calculate_rouge,
    flatten_list,
    freeze_embeds,
    freeze_params,
    label_smoothed_nll_loss,
    lmap,
    save_json,
)

from src.models.thu_coai_hint.lightning_base import BaseTransformer

logger = logging.getLogger(__name__)


class SummarizationModel(BaseTransformer):
    mode = "summarization"
    loss_names = ["loss", "lm_loss", "reorder_loss", "sbert_loss"]
    metric_names = []
    default_val_metric = "loss"

    def __init__(self, hparams, **kwargs):
        self._check_hparams(hparams)
        super().__init__(hparams, model_type=self.mode, **kwargs)

        # self.tokenizer.add_special_tokens({"additional_special_tokens": ["<sen>"]})
        # self.model.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
        self.metrics_save_path = Path(self.experiment_output_dir) / "metrics.json"
        self.step_count = 0
        self.metrics = defaultdict(list)
        self.current_val_metrics = None
        self.model_type = self.config.model_type
        self.vocab_size = len(self.tokenizer)

        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            prefix=self.model.config.prefix or "",
        )
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length,
        }
        assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"
        # Whether changing embeddings
        if self.hparams.freeze_embeds:
            freeze_embeds(self.model)
        if self.hparams.freeze_encoder:
            freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())

        self.num_workers = hparams.num_workers
        self.decoder_start_token_id = self.model.config.decoder_start_token_id  # default to config
        # compatible to MBartTokenizer
        if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, MBartTokenizer):
            self.decoder_start_token_id = self.tokenizer.lang_code_to_id[hparams.tgt_lang]
            self.model.config.decoder_start_token_id = self.decoder_start_token_id

        self.dataset_class = Seq2SeqSbertscoreOrderDisDataset

        self.already_saved_batch = False
        self.eval_beams = self.model.config.num_beams if self.hparams.eval_beams is None else self.hparams.eval_beams
        if self.hparams.eval_max_gen_length is not None:
            self.eval_max_length = self.hparams.eval_max_gen_length
        else:
            self.eval_max_length = self.model.config.max_length
        self.val_metric = self.default_val_metric if self.hparams.val_metric is None else self.hparams.val_metric

        if hparams.reorder:
            self.reorder_linear_layer = torch.nn.Linear(self.config.d_model, self.config.d_model, bias=True)
        if hparams.sbert:
            self.sbert_linear_layer = torch.nn.Linear(self.config.d_model, self.config.d_model, bias=True)

    def _check_hparams(self, hparams):
        if hparams.sortish_sampler and hparams.gpus > 1:
            hparams.replace_sampler_ddp = False
        elif hparams.max_tokens_per_batch is not None:
            if self.hparams.gpus > 1:
                raise NotImplementedError("Dynamic Batch size does not work for multi-gpu training")
            if hparams.sortish_sampler:
                raise ValueError("--sortish_sampler and --max_tokens_per_batch may not be used simultaneously")

    def save_readable_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        """A debugging utility"""
        readable_batch = {
            k: self.tokenizer.batch_decode(v.tolist()) if "mask" not in k else v.shape for k, v in batch.items()
        }
        save_json(readable_batch, Path(self.experiment_output_dir) / "text_batch.json")
        save_json({k: v.tolist() for k, v in batch.items()}, Path(self.experiment_output_dir) / "tok_batch.json")

        self.already_saved_batch = True
        return readable_batch

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def gather_nd(self, x, indices):
        newshape = indices.shape[:-1] + x.shape[indices.shape[-1]:]
        indices = indices.view(-1, indices.shape[-1]).tolist()
        out = torch.cat([x.__getitem__(tuple(i)) for i in indices]).reshape(newshape)
        return out

    def _step(self, batch: dict) -> Tuple:
        hparams = self.hparams

        pad_token_id = self.tokenizer.pad_token_id
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        tgt_ids = batch["labels"]
        if isinstance(self.model, T5ForConditionalGeneration):
            decoder_input_ids = self.model._shift_right(tgt_ids)
        else:
            decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id, self.decoder_start_token_id)
        if self.hparams.save_readable_batch and not self.already_saved_batch:
            # This would be slightly better if it only happened on rank zero
            batch["decoder_input_ids"] = decoder_input_ids
            self.save_readable_batch(batch)
        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False,
                       output_attentions=True, output_hidden_states=True)

        sen_token_id = self.tokenizer.mask_token_id

        lm_logits = outputs["logits"]

        if self.hparams.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            # [batch_size*sequence_length]
            loss_mask = (batch["normal_labels"][:, None] * torch.ones_like(tgt_ids) * (
                    1 - tgt_ids.eq(pad_token_id).to(torch.float))).view(-1)

            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction="none")
            assert lm_logits.shape[-1] == self.vocab_size
            loss = torch.sum(ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1)) * loss_mask) / (
                    torch.sum(loss_mask) + 1e-20)
        else:
            lprobs = torch.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, tgt_ids, self.hparams.label_smoothing, ignore_index=pad_token_id
            )
        lm_loss = loss + 0.
        reorder_loss, sbert_loss = torch.tensor(0., ), torch.tensor(0., )

        # [batch_size, sequence_length, hidden_size]
        hidden_states = outputs["decoder_hidden_states"][-1]
        batch_size, sequence_length, hidden_size = hidden_states.size()
        # [batch_size, sequence_length]
        sen_pos = batch["labels"].eq(sen_token_id).to(torch.float)
        dis_pos = torch.cat([torch.zeros([batch_size, 1]).to(sen_pos.device), sen_pos[:, :-1]], 1)
        # coeff = 1. / (float(hparams.reorder) + float(hparams.sbert))

        if hparams.reorder:
            sen_idx = dis_pos.nonzero()

            # [bath_size, sen_length]
            reorder_label = batch["orders"]
            sentence_num = reorder_label.size()[1]

            # [batch_size, sentence_num, hidden_size]
            sent_hidden_states_gather = self.gather_nd(hidden_states, sen_idx)
            sent_hidden_states = torch.reshape(sent_hidden_states_gather, [batch_size, sentence_num, hidden_size])

            self.reorder_linear_layer = self.reorder_linear_layer.float()

            # [batch_size, sentence_num, sentence_num]
            sen_att_logits = torch.matmul(self.reorder_linear_layer(sent_hidden_states),
                                          torch.transpose(sent_hidden_states, 1, 2))

            # [batch_size, sentence_num, sentence_num]
            pred_score = torch.sigmoid(sen_att_logits)  # + torch.transpose(sen_att_logits, 1, 2))
            reorder_mask = (1 - torch.eye(sentence_num)[None, :, :].to(batch["type_labels"].device)) * \
                           (torch.eq(batch["type_labels"], 0) | torch.eq(batch["type_labels"], 1)).to(torch.float)[:,
                           None, None]

            true_label = torch.arange(sentence_num)[None, :].to(reorder_label.device)
            true_label_matrix = torch.lt(true_label[:, :, None], true_label[:, None, :]).to(torch.float)
            reorder_mask *= true_label_matrix
            reorder_mask *= (1 - torch.lt(true_label[:, :, None] + 2, true_label[:, None, :]).to(torch.float))

            tmp_minus = reorder_label[:, None, :] - reorder_label[:, :, None]
            reorder_label_matrix = (torch.lt(tmp_minus, 3) & torch.lt(-tmp_minus, 0)).to(torch.float)
            batch_reorder_loss = -torch.log(pred_score + 1e-20) * reorder_label_matrix - torch.log(
                1 - pred_score + 1e-20) * (1 - reorder_label_matrix)

            batch_reorder_loss *= reorder_mask
            reorder_loss = torch.mean(
                torch.sum(batch_reorder_loss, [1, 2]) / (torch.sum(reorder_mask, [1, 2]) + 1e-20) * (
                        1 - 0.99 * batch["normal_labels"]))
            loss += reorder_loss

        if hparams.sbert:
            sen_idx = sen_pos.nonzero()

            # [bath_size, sentence_num, sentence_num]
            sbert_score_label = batch["sbert_score"]
            sentence_num = sbert_score_label.size()[1]

            # [batch_size, sentence_num, hidden_size]
            sent_hidden_states_gather = self.gather_nd(hidden_states, sen_idx)
            sent_hidden_states = torch.reshape(sent_hidden_states_gather, [batch_size, sentence_num, hidden_size])
            self.sbert_linear_layer = self.sbert_linear_layer.float()

            # [batch_size, sentence_num, sentence_num]
            pred = torch.matmul(self.sbert_linear_layer(sent_hidden_states), torch.transpose(sent_hidden_states, 1, 2))
            pred_score = -1 + 2 * torch.sigmoid(pred + torch.transpose(pred, 1, 2))

            true_label = torch.arange(sentence_num)[None, :].to(pred.device)
            true_label_matrix = torch.le(true_label[:, :, None], true_label[:, None, :]).to(torch.float)
            sbert_mask = torch.ones_like(pred_score)

            batch_sbert_loss = torch.max(torch.abs(pred_score - sbert_score_label) - 0.1,
                                         torch.zeros_like(pred_score).to(pred_score.device))
            batch_sbert_loss *= sbert_mask
            sbert_loss = 0.1 * torch.sum(batch_sbert_loss) / (torch.sum(sbert_mask) + 1e-20)
            loss += sbert_loss

        return (loss, lm_loss, reorder_loss, sbert_loss)

    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch)
        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        # metrics logged can be access by trainer.callback_metrics
        self.log_dict(self.current_val_metrics)
        # tokens per batch
        logs["tpb"] = batch["input_ids"].ne(self.pad).sum() + batch["labels"].ne(self.pad).sum()
        logs["bs"] = batch["input_ids"].shape[0]
        logs["src_pad_tok"] = batch["input_ids"].eq(self.pad).sum()
        logs["src_pad_frac"] = batch["input_ids"].eq(self.pad).float().mean()
        # TODO(SS): make a wandb summary metric for this
        return {"loss": loss_tensors[0], "log": logs}

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses["loss"]
        generative_metrics = {
            k: np.array([x[k] for x in outputs]).mean() for k in self.metric_names + ["gen_time", "gen_len"]
        }
        metric_val = (
            torch.tensor(generative_metrics[self.val_metric])
            if self.val_metric in generative_metrics else losses[self.val_metric]
        )
        metric_tensor: torch.FloatTensor = metric_val.type_as(loss)
        generative_metrics.update({k: v.item() for k, v in losses.items()})
        losses.update(generative_metrics)
        all_metrics = {f"{prefix}_{k}": x for k, x in losses.items()}
        all_metrics["step_count"] = self.step_count
        self.metrics[prefix].append(all_metrics)  # callback writes this to self.metrics_save_path.
        self.current_val_metrics = all_metrics
        print(f"Evaluation result: {all_metrics}")
        preds = flatten_list([x["preds"] for x in outputs])
        return {
            "log": all_metrics,
            "preds": preds,
            f"{prefix}_loss": loss,
            f"{prefix}_{self.val_metric}": metric_tensor,
        }

    def calc_generative_metrics(self, preds, target) -> Dict:
        return calculate_rouge(preds, target)

    def _generative_step(self, batch: dict) -> dict:
        t0 = time.time()

        generated_ids = self.model.model_generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
            decoder_start_token_id=self.decoder_start_token_id,
            num_beams=self.eval_beams,
            max_length=self.eval_max_length,
        )
        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        target: List[str] = self.ids_to_clean_text(batch["labels"])
        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        rouge: Dict = self.calc_generative_metrics(preds, target)
        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds, target=target, **rouge)
        return base_metrics

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")

    def get_dataset(self, type_path) -> Seq2SeqDataset:
        n_obs = self.n_obs[type_path]
        max_target_length = self.target_lens[type_path]
        dataset = self.dataset_class(
            self.tokenizer,
            type_path=type_path,
            n_obs=n_obs,
            max_target_length=max_target_length,
            **self.dataset_kwargs,
        )
        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(type_path)

        if self.hparams.sortish_sampler and type_path != "test" and type_path != "val":
            sampler = dataset.make_sortish_sampler(batch_size, distributed=self.hparams.n_gpu > 1)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=False,
                num_workers=self.num_workers,
                sampler=sampler,
            )

        elif self.hparams.max_tokens_per_batch is not None and type_path != "test" and type_path != "val":
            batch_sampler = dataset.make_dynamic_sampler(
                self.hparams.max_tokens_per_batch, distributed=self.hparams.n_gpu > 1
            )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=dataset.collate_fn,
                shuffle=False,
                num_workers=self.num_workers,
                batch_size=batch_size,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=shuffle,
                num_workers=self.num_workers,
                sampler=None,
            )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size, shuffle=False)


class TranslationModel(SummarizationModel):
    mode = "translation"
    loss_names = ["loss"]
    metric_names = ["bleu"]
    default_val_metric = "bleu"

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.dataset_kwargs["src_lang"] = hparams.src_lang
        self.dataset_kwargs["tgt_lang"] = hparams.tgt_lang

    def calc_generative_metrics(self, preds, target) -> dict:
        return calculate_bleu(preds, target)
