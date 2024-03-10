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
import openai
import concurrent.futures
import os
from retrying import retry

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from transformers import BartTokenizer

from src.configuration.event_trigger.config_args import parse_args_for_config
from src.utils.file_utils import copy_file_or_dir, output_obj_to_file
from src.utils import nlg_eval_utils
from tasks.cetrica.chatglm_generate import ChatglmGenerator


class ChatgptGenerator(ChatglmGenerator):
    def __init__(self):
        # parameters
        super().__init__()
        openai.api_key = os.getenv("CHAT_API")
        self.max_workers = 10
        self.data_limit = 500

    def revise_generate(self, dataset_name="roc-stories", model_name="gpt-3.5-turbo", revised_dest_name="event-lm-sbert-roc-stories"):
        """
        Revise the generated stories
        """
        assert dataset_name in ["roc-stories", "writing-prompts"]
        assert model_name in ["gpt-3.5-turbo", "gpt-3.5", "gpt-4"]
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


        leadings = self.read_list(self.src_file)[:self.data_limit]
        events = self.read_list(self.event_file)[:self.data_limit]
        origin_stories = self.read_list(self.waiting_to_revise_file)[:self.data_limit]
        input_texts = self.compose_prompts("revise", dataset_name, leadings, events, origin_stories=origin_stories)[:self.data_limit]
        tgt_stories = self.read_list(self.tgt_file)[:self.data_limit]

        copy_file_or_dir(self.src_file, self.generation_dir / "test.source.txt")
        copy_file_or_dir(self.event_file, self.generation_dir / "test_event.source.txt")
        copy_file_or_dir(self.tgt_file, self.generation_dir / "test.target.txt")
        self.output_list(input_texts, self.generation_dir / f"revise_prompts.source.txt")

        generated_stories = self.chatgpt_batch_func_with_cache(input_texts, self.gen_file, self.max_workers)
        metrics = self.eval_output(pred_lines=generated_stories, tgt_lines=tgt_stories)
        output_obj_to_file(json.dumps(metrics, indent=4), self.eval_file)

    def model_generate(self, mode="leading_event", dataset_name="roc-stories", model_name="gpt-3.5-turbo"):
        """
        generate stories according to given conditions
        """
        assert dataset_name in ["roc-stories", "writing-prompts"]
        assert model_name in ["gpt-3.5-turbo", "gpt-3.5", "gpt-4"]
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


        leadings = self.read_list(self.src_file)[:self.data_limit]
        events = self.read_list(self.event_file)[:self.data_limit]
        input_texts = self.compose_prompts(mode, dataset_name, leadings, events)[:self.data_limit]
        tgt_stories = self.read_list(self.tgt_file)[:self.data_limit]

        copy_file_or_dir(self.src_file, self.generation_dir / "test.source.txt")
        copy_file_or_dir(self.event_file, self.generation_dir / "test_event.source.txt")
        copy_file_or_dir(self.tgt_file, self.generation_dir / "test.target.txt")
        self.output_list(input_texts, self.generation_dir / f"{mode}_prompts.source.txt")

        generated_stories = self.chatgpt_batch_func_with_cache(input_texts, self.gen_file, self.max_workers)
        metrics = self.eval_output(pred_lines=generated_stories, tgt_lines=tgt_stories)
        output_obj_to_file(json.dumps(metrics, indent=4), self.eval_file)

    def chatgpt_batch_func_with_cache(self, input_texts: list, tgt_file: Path, max_workers=512):
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
            # for input_text in tqdm(input_texts[len_results:], desc="chatgpt_func_with_cache"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 使用map方法，将输入数据分配给不同的线程，并获取返回值
                print(f"使用executor开启多线程; max_workers:{max_workers}")
                data = input_texts[len_results:]
                thread_results = executor.map(self.chatgpt_func,
                                              data,
                                              [i for i in range(len(data))],
                                              [len(data)]*len(data))
                for response in thread_results:
                    response = response.replace("\n", "") + "\n"
                    fw.write(response)
                    results.append(response)
                    fw.flush()
        return results

    @retry(stop_max_attempt_number=10, wait_exponential_multiplier=1000, wait_exponential_max=10000)
    def chatgpt_func(self, input_text: str, data_idx=None, total_data_size=None):
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
            {"role": "user", "content": json.dumps(input_text)}])
        response = completion.choices[0].message.content
        if data_idx is not None:
            print(f"current progress: data_idx: {data_idx}; total_data_size:{total_data_size}")
        return f"{response}\n"


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
    chatgpt = ChatgptGenerator()
    chatgpt.max_workers = 4
    chatgpt.data_limit = 300
    # generate predicted stories
    # chatgpt.model_generate(mode="leading_event", dataset_name="roc-stories", model_name="gpt-3.5-turbo")
    # chatgpt.model_generate(mode="leading_event", dataset_name="writing-prompts", model_name="gpt-3.5-turbo")
    chatgpt.revise_generate(dataset_name="roc-stories", model_name="gpt-3.5-turbo", revised_dest_name="event-lm-sbert-roc-stories")
    chatgpt.revise_generate(dataset_name="writing-prompts", model_name="gpt-3.5-turbo", revised_dest_name="event-lm-sbert-writing-prompts")
