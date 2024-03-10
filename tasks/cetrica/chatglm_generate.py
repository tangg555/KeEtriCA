"""
@Desc:
@Reference:
@Notes:
WANDB is Weights and Biases Logger:
https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.wandb.html
"""

import sys
import json
import numpy as np
from pathlib import Path
import requests
from tqdm import tqdm
import os
from retrying import retry

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from transformers import BartTokenizer

from src.configuration.event_trigger.config_args import parse_args_for_config
from src.utils.file_utils import copy_file_or_dir, output_obj_to_file
from src.utils import nlg_eval_utils

class ChatglmGenerator(object):
    def __init__(self):
        # parameters
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.generation_dir = None
        self.gen_file = None
        self.eval_file = None
        self.dataset_dir = None

        self.src_file = None
        self.event_file = None
        self.tgt_file = None
        self.gen_file = None

    def output_list(self, list_obj: list, file_path: Path):
        with file_path.open("w", encoding="utf-8") as fw:
            fw.writelines([one.replace('\n', '') + "\n" for one in list_obj])

    def read_list(self, file_path: Path):
        with file_path.open("r", encoding="utf-8") as fr:
            lines = fr.readlines()
        return [one.strip() for one in lines]

    def limit_words(self, content:str, max_len):
        words = content.strip().split()
        return " ".join(words[:max_len])

    def compose_prompts(self, mode, dataset_name, leadings, events, origin_stories=None):
        input_texts = []
        assert dataset_name in ["roc-stories", "writing-prompts"]
        length_of_stories = 4 if dataset_name == "roc-stories" else 10
        if mode == "leading_event":
            for leading, event_sequence in zip(leadings, events):
                events = event_sequence.replace("[EVENT_s]", "").replace("[EVENT_e]", "").strip().split(" [EVENT_sep] ")
                input_text = f'Given a leading context as the first sentence and a sequence of events, ' \
                             f'the task is to compose a {length_of_stories}-sentence story in English excluding the leading context.' \
                             f' The character names will be represented by "[MALE]", "[FEMALE]" and "[NEUTRAL]". ' \
                             f'Each event, separated by the comma, must be incorporated into each sentence. ' \
                             f'The resulting story should be directly outputted without redundancy. ' \
                             f'\n\nThe Leading Context:\n' \
                             f'{leading}\n\n' \
                             f'Events:\n' \
                             f'{", ".join(events)}\n\n' \
                             f'Output:'
                input_text = json.dumps(input_text)
                input_texts.append(input_text)
        elif mode == "revise":
            assert len(leadings) == len(events) and len(leadings) == len(origin_stories)
            for leading, event_sequence, origin_story in zip(leadings, events, origin_stories):
                events = event_sequence.replace("[EVENT_s]", "").replace("[EVENT_e]", "").strip().split(" [EVENT_sep] ")
                input_text = f'Given a leading context as the first sentence and a sequence of events, ' \
                             f'the task is to revise a given {length_of_stories}-sentence story in English excluding the leading context as the reference.' \
                             f' The character names will be represented by "[MALE]", "[FEMALE]" and "[NEUTRAL]". ' \
                             f'Each event, separated by the comma, must be incorporated into each sentence. ' \
                             f'The resulting story should be directly outputted without redundancy. ' \
                             f'\n\nThe Leading Context:\n' \
                             f'{leading}\n\n' \
                             f'Events:\n' \
                             f'{", ".join(events)}\n\n' \
                             f'The Reference Story:\n' \
                             f'{origin_story.strip()}\n\n' \
                             f'Output:'
                input_text = json.dumps(input_text)
                input_texts.append(input_text)
        else:
            raise NotImplementedError()
        return input_texts

    def revise_generate(self, dataset_name="roc-stories", model_name="chatglm-6b", revised_dest_name="event-lm-sbert-roc-stories"):
        """
        Revise the generated stories
        """
        assert dataset_name in ["roc-stories", "writing-prompts"]
        assert model_name in ["chatglm-6b", "chatglm-10b"]
        self.generation_dir = Path(f"{BASE_DIR}/output/cetrica/{model_name}_{dataset_name}_revise_gen_result")
        self.generation_dir.mkdir(parents=True, exist_ok=True)
        self.gen_file = self.generation_dir / f"revise_gen.txt"
        self.eval_file = self.generation_dir / f"revise_eval.txt"
        self.waiting_to_revise_file = Path(f"{BASE_DIR}/output/cetrica/{revised_dest_name}/gen_result/test_event.source_gen.txt")
        if dataset_name == "roc-stories":
            self.dataset_dir = Path(f"{BASE_DIR}/datasets/event-trigger/roc-stories")
        elif dataset_name == "writing-prompts":
            self.dataset_dir = Path(f"{BASE_DIR}/datasets/event-trigger/writing-prompts")
        else:
            raise NotImplementedError()

        self.src_file = self.dataset_dir / "test.source.txt"
        self.event_file = self.dataset_dir / "test_event.source.txt"
        self.tgt_file = self.dataset_dir / "test.target.txt"
        print(f"mode: {model_name}_{dataset_name}_revise generating")
        print(f"src_file: {self.src_file}\n"
              f"event_file: {self.event_file}\n"
              f"tgt_file: {self.tgt_file}\n"
              f"gen_file: {self.gen_file}\n"
              f"waiting_to_revise_file: {self.waiting_to_revise_file}\n")


        leadings = self.read_list(self.src_file)
        events = self.read_list(self.event_file)
        origin_stories = self.read_list(self.waiting_to_revise_file)
        input_texts = self.compose_prompts("revise", dataset_name, leadings, events, origin_stories=origin_stories)
        tgt_stories = self.read_list(self.tgt_file)

        copy_file_or_dir(self.src_file, self.generation_dir / "test.source.txt")
        copy_file_or_dir(self.event_file, self.generation_dir / "test_event.source.txt")
        copy_file_or_dir(self.tgt_file, self.generation_dir / "test.target.txt")
        self.output_list(input_texts, self.generation_dir / f"revise_prompts.source.txt")

        generated_stories = self.chatglm_batch_func_with_cache(input_texts, self.gen_file)
        metrics = self.eval_output(pred_lines=generated_stories, tgt_lines=tgt_stories)
        output_obj_to_file(json.dumps(metrics, indent=4), self.eval_file)

    def model_generate(self, mode="leading_event", dataset_name="roc-stories", model_name="chatglm-6b"):
        """
        generate stories according to given conditions
        """
        assert dataset_name in ["roc-stories", "writing-prompts"]
        assert model_name in ["chatglm-6b", "chatglm-10b"]
        self.generation_dir = Path(f"{BASE_DIR}/output/cetrica/{model_name}_{dataset_name}_{mode}_gen_result")
        self.generation_dir.mkdir(parents=True, exist_ok=True)
        self.gen_file = self.generation_dir / f"{mode}_gen.txt"
        self.eval_file = self.generation_dir / f"{mode}_eval.txt"
        self.dataset_dir = Path(f"{BASE_DIR}/datasets/event-trigger/{dataset_name}")

        self.src_file = self.dataset_dir / "test.source.txt"
        self.event_file = self.dataset_dir / "test_event.source.txt"
        self.tgt_file = self.dataset_dir / "test.target.txt"
        print(f"mode: {model_name}_{dataset_name}_{mode} generating")
        print(f"src_file: {self.src_file}\n"
              f"event_file: {self.event_file}\n"
              f"tgt_file: {self.tgt_file}\n"
              f"gen_file: {self.gen_file}\n")


        leadings = self.read_list(self.src_file)
        events = self.read_list(self.event_file)
        input_texts = self.compose_prompts(mode, dataset_name, leadings, events)
        tgt_stories = self.read_list(self.tgt_file)

        copy_file_or_dir(self.src_file, self.generation_dir / "test.source.txt")
        copy_file_or_dir(self.event_file, self.generation_dir / "test_event.source.txt")
        copy_file_or_dir(self.tgt_file, self.generation_dir / "test.target.txt")
        self.output_list(input_texts, self.generation_dir / f"{mode}_prompts.source.txt")

        generated_stories = self.chatglm_batch_func_with_cache(input_texts, self.gen_file)
        metrics = self.eval_output(pred_lines=generated_stories, tgt_lines=tgt_stories)
        output_obj_to_file(json.dumps(metrics, indent=4), self.eval_file)

    def chatglm_func(self, input_text: str, retry_time_count=0):
        headers = {"Content-Type": "application/json"}
        data_dict = {"prompt": f'\"{json.dumps(input_text)}\"', "history": []}
        data = json.dumps(data_dict)
        url = "https://u7705-adaa-7cbf49af.neimeng.seetacloud.com:6443"
        res_obj = requests.post(url=url, headers=headers, data=data)
        try:
            response = res_obj.json()['response']
        except Exception as e:
            if retry_time_count > 3:
                raise ValueError(f"Fail to retry. retry_time_count: {retry_time_count}")
            print(f"||||||||||||original input: {res_obj.text}")
            print(f"||||||||||||Exception: {e}")
            print(f"||||||||||||retry ({retry_time_count+1} times):")
            response = self.chatglm_func(input_text, retry_time_count=retry_time_count+1)
        return f"{response}\n"


    def chatglm_batch_func(self, input_texts: list):
        resutls = []
        for one in tqdm(input_texts, desc="chatgpt_func"):
            origin_text = self.chatglm_func(one)
            resutls.append(origin_text.replace("\n", "") + "\n")
        return resutls

    def clean_generated_results(self, input_texts, results):
        cleaned_results = []
        count = 0
        for line in tqdm(results, desc="clean_chatgpt_results"):
            entry_to_rerun = "Unable to fetch the response, Please try again."
            if line.strip() == entry_to_rerun.strip() or line.strip() == "":
                response = self.chatglm_func(input_texts[count])
                response = response.replace("\n", "") + "\n"
                cleaned_results.append(response)
            else:
                cleaned_results.append(line)
            count += 1
        return cleaned_results

    def chatglm_batch_func_with_cache(self, input_texts: list, tgt_file: Path):
        results = []
        if tgt_file.exists():
            with tgt_file.open("r", encoding="utf-8") as fr:
                results = fr.readlines()
        results = self.clean_generated_results(input_texts, results)
        len_results = len(results)
        print(f"{len_results} has already been generated.")
        with tgt_file.open("w", encoding="utf-8") as fw:
            fw.writelines(results)
            fw.flush()
            for input_text in tqdm(input_texts[len_results:], desc="chatgglm_func_with_cache"):
                response = self.chatglm_func(input_text)
                response = response.replace("\n", "") + "\n"
                fw.write(response)
                results.append(response)
                fw.flush()
        return results

    def eval_output(self, pred_lines: list, tgt_lines: list):
        tgt_lines_toks, pred_lines_toks = \
            [self.tokenizer.tokenize(t) for t in tgt_lines], [self.tokenizer.tokenize(c) for c in pred_lines]

        metrics = {}
        nlg_eval_utils.calculate_bleu(ref_lines=tgt_lines_toks, gen_lines=pred_lines_toks, metrics=metrics)
        # calculate rouge score
        rouge_metrics = nlg_eval_utils.calculate_rouge(pred_lines=pred_lines, tgt_lines=tgt_lines)
        metrics.update(**rouge_metrics)
        # calculate repetition and distinction
        nlg_eval_utils.repetition_distinction_metric(pred_lines_toks, metrics=metrics, repetition_times=2)
        key = sorted(metrics.keys())
        for k in key:
            print(k, metrics[k])
        return metrics

if __name__ == '__main__':
    chatglm = ChatglmGenerator()

    # generate predicted stories
    chatglm.model_generate(mode="leading_event", dataset_name="roc-stories", model_name="chatglm-6b")
    chatglm.model_generate(mode="leading_event", dataset_name="writing-prompts", model_name="chatglm-6b")
    # chatglm.revise_generate(dataset_name="roc-stories", model_name="chatglm-6b", revised_dest_name="event-lm-sbert-roc-stories")
    # chatglm.revise_generate(dataset_name="writing-prompts", model_name="chatglm-6b", revised_dest_name="event-lm-sbert-writing-prompts")
