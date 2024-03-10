"""
@Desc:
@Reference:
- from_argparse_args:
https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.utilities.argparse.html#p
ytorch_lightning.utilities.argparse.from_argparse_args
- ModelCheckpoint
https://pytorch-lightning.readthedocs.io/en/latest/extensions/generated/
pytorch_lightning.callbacks.ModelCheckpoint.html?highlight=ModelCheckpoint
-  Trainer.from_argparse_args(args)
https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
- Optimizers: AdamW and AdaFactor
https://huggingface.co/docs/transformers/main_classes/optimizer_schedules
Adaptive Learning Rates with Sublinear Memory Cost https://arxiv.org/abs/1804.04235
- optimizer_grouped_parameters
https://huggingface.co/transformers/v3.3.1/training.html
The optimizer allows us to apply different hyperpameters for specific parameter groups.
- rank_zero_only
http://www.liuxiao.org/2020/07/pytorch-lighting-%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E6%95%B4%E7%90%86/
使用 @rank_zero_only 修饰多线程中只在 RANK=0 调用的函数
@Notes:
model.prepare_data()
initialize_distributed()
model.setup(stage)
model.train_dataloader()
model.val_dataloader()
model.test_dataloader()
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info

from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
)

from transformers.optimization import (
    Adafactor,
)

from src.configuration.constants import MODEL_CLASSES, GET_SCHEDULER_FUNCS, GET_SCHEDULER_FUNC_CHOICES
from src.utils.string_utils import are_same_strings

logger = logging.getLogger(__name__)


class BaseTransformer(pl.LightningModule):
    def __init__(
        self,
        hparams: argparse.Namespace,
        model_type="base",
        config: PretrainedConfig=None,
        tokenizer: PreTrainedTokenizer=None,
        model=None,
        **config_kwargs
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        self.save_hyperparameters(hparams)  # save hparams to self.hparams
        self.output_dir = Path(self.hparams.output_dir)
        self.experiment_output_dir = os.path.join(self.output_dir, self.hparams.experiment_name)
        # load pretrained settings
        # config
        self.config: PretrainedConfig = config if config is not None else \
            AutoConfig.from_pretrained(self.hparams.model_name_or_path,
                                       **config_kwargs)
        self._check_config(self.config)
        # tokenizer
        self.tokenizer: PreTrainedTokenizer = tokenizer if tokenizer is not None else \
            AutoTokenizer.from_pretrained(self.hparams.model_name_or_path)
        # model
        self.model_class = MODEL_CLASSES[model_type]
        self.model = model if model is not None\
            else self._load_model(self.hparams.model_name_or_path, self.model_class, config)

        # record api
        self.optimizer = None
        self.scheduler = None

    def _check_config(self, config: PretrainedConfig):
        extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
        for p in extra_model_params:
            if getattr(self.hparams, p, None):
                assert hasattr(config, p), f"model config doesn't have a `{p}` attribute"
                setattr(config, p, getattr(self.hparams, p))

    def _load_model(self, model_name_or_path: str, model_class, config: PretrainedConfig=None):
        if config is None:
            return model_class.from_pretrained(
                model_name_or_path,
            )
        else:
            return model_class.from_pretrained(
                model_name_or_path,
                config=config,
            )

    def get_lr_scheduler(self, optimizer: torch.optim.Optimizer):
        get_schedule_func = GET_SCHEDULER_FUNCS[self.hparams.lr_scheduler]
        scheduler = get_schedule_func(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps()
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if are_same_strings(self.hparams.optimizer_class, "Adafactor"):
            optimizer = Adafactor(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False, relative_step=False
            )

        elif are_same_strings(self.hparams.optimizer_class, "AdamW"):
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
            )
        else:
            raise NotImplementedError(f"{self.hparams.optimizer_class} not available. Only Adafactor|Adafactor")

        self.optimizer = optimizer

        self.scheduler = self.get_lr_scheduler(self.optimizer)

        return [self.optimizer], [self.scheduler]

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        return self.validation_end(outputs)

    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        if isinstance(self.hparams.accumulate_grad_batches, dict):
            accumulate_grad_batches = list(self.hparams.accumulate_grad_batches.values())[-1]
        else:
            accumulate_grad_batches = self.hparams.accumulate_grad_batches
        effective_batch_size = self.hparams.train_batch_size * accumulate_grad_batches * num_devices
        return (self.dataset_size / effective_batch_size) * self.hparams.max_epochs

    def setup(self, stage: Optional[str] = None):
        if stage == "test":
            self.dataset_size = len(self.test_dataloader().dataset)
        else:
            self.train_loader = self.get_dataloader("train", self.hparams.train_batch_size, shuffle=True)
            self.dataset_size = len(self.train_dataloader().dataset)

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False):
        raise NotImplementedError("You must implement this for your task")

    def train_dataloader(self):
        return self.get_dataloader("train", batch_size=self.hparams.eval_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", batch_size=self.hparams.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size, shuffle=False)

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = Path(self.experiment_output_dir).joinpath("best_tfmr")
        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

