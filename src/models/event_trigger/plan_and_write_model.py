"""
@Desc:
@Reference:
- Plan-And-Write: Towards Better Automatic Storytelling
https://arxiv.org/pdf/1811.05701.pdf
@Notes:
"""

import logging
from typing import Dict, List, Tuple
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers.models.bart.modeling_bart import BartForConditionalGeneration, BartConfig
from transformers import BartTokenizer
from transformers.models.bart import modeling_bart

from src.utils.gen_utils import ids_to_clean_string, top_p_logits
from src.modules.event_trigger.datasets import (
    LeadingContextDataset,
    EventLineDataset,
    LeadingToEventsDataset,
    LeadingPlusEventDataset
)
from src.models.event_trigger import (
    LeadingContextBart,
    EventBart,
    LeadingToEventsBart,
    LeadingPlusEventBart,
)
from src.modules.event_trigger.plan_and_write_modules import PlanAWForCG

logger = logging.getLogger(__name__)


class LeadingContextPlanAW(LeadingContextBart):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

    def _custom_init(self):
        # load pretrained settings from bart
        # config
        self.config: BartConfig = BartConfig.from_pretrained(self.hparams.model_name_or_path)
        # tokenizer
        self.tokenizer: BartTokenizer = BartTokenizer.from_pretrained(self.hparams.model_name_or_path)
        # model
        self.model = self._load_model(self.hparams.model_name_or_path, PlanAWForCG, self.config)
        self._set_up(config=self.config,
                     tokenizer=self.tokenizer,
                     model=self.model)
        self.dataset_class = LeadingContextDataset


class EventPlanAW(EventBart):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

    def _custom_init(self):
        # load pretrained settings from bart
        # config
        self.config: BartConfig = BartConfig.from_pretrained(self.hparams.model_name_or_path)
        # tokenizer
        self.tokenizer: BartTokenizer = BartTokenizer.from_pretrained(self.hparams.model_name_or_path)
        # model
        self.model = self._load_model(self.hparams.model_name_or_path, PlanAWForCG, self.config)
        self._set_up(config=self.config,
                     tokenizer=self.tokenizer,
                     model=self.model)
        self.dataset_class = EventLineDataset

        self.train_event_infix = "_event"
        self.test_event_infix = self.hparams.test_event_infix if self.hparams.test_event_infix else "_predicted_event"
        self.eval_event_infix = self.test_event_infix


class LeadingToEventsPlanAW(LeadingToEventsBart):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

    def _custom_init(self):
        # load pretrained settings from bart
        # config
        self.config: BartConfig = BartConfig.from_pretrained(self.hparams.model_name_or_path)
        # tokenizer
        self.tokenizer: BartTokenizer = BartTokenizer.from_pretrained(self.hparams.model_name_or_path)
        # model
        self.model = self._load_model(self.hparams.model_name_or_path, PlanAWForCG, self.config)
        self._set_up(config=self.config,
                     tokenizer=self.tokenizer,
                     model=self.model)
        self.dataset_class = LeadingToEventsDataset
        self.train_event_infix = "_event"
        self.test_event_infix = self.hparams.test_event_infix if self.hparams.test_event_infix else "_predicted_event"
        self.eval_event_infix = self.test_event_infix


class LeadingPlusEventPlanAW(LeadingPlusEventBart):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

    def _custom_init(self):
        # load pretrained settings from bart
        # config
        self.config: BartConfig = BartConfig.from_pretrained(self.hparams.model_name_or_path)
        # tokenizer
        self.tokenizer: BartTokenizer = BartTokenizer.from_pretrained(self.hparams.model_name_or_path)
        # model
        self.model = self._load_model(self.hparams.model_name_or_path, PlanAWForCG, self.config)
        self._set_up(config=self.config,
                     tokenizer=self.tokenizer,
                     model=self.model)
        self.dataset_class = LeadingPlusEventDataset

        self.train_event_infix = "_event"
        self.test_event_infix = self.hparams.test_event_infix if self.hparams.test_event_infix else "_predicted_event"
        self.eval_event_infix = self.test_event_infix
