"""
@Desc:
@Reference:
@Notes:
"""

import sys
import os
from shutil import copyfile
from pathlib import Path

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from src.utils.event_trigger.stat_utils import parse_files, parse_event_graphs
from preprocessing.event_trigger.event_predictor import EventPredictor
from src.utils.event_trigger.event_analyzer_utils import EventAnalyzer


def get_events(event_line: str):
    elements = event_line.strip().split()[1:-1]
    events = []
    for one in elements:
        if one == "[EVENT_sep]":
            continue
        events.append(one)
    return events


def parse_stat(src_file: Path, tgt_file: Path, event_file: Path):
    src_lines = src_file.open("r", encoding="utf-8").readlines()
    tgt_lines = tgt_file.open("r", encoding="utf-8").readlines()
    event_lines = event_file.open("r", encoding="utf-8").readlines()

    stats = {
        "stories": 0,
        "vocab": 0,
        "events set": 0,
        "avg. input words": 0,
        "avg. output words": 0,
        "avg. events": 0,
    }
    vocab = set()
    event_set = set()
    for src_line, tgt_line, event_line in zip(src_lines, tgt_lines, event_lines):
        stats["stories"] += 1
        src_words = [one for one in src_line.strip().split()]
        tgt_words = [one for one in tgt_line.strip().split()]
        event_words = [one for one in event_line.strip().split()]
        vocab.update(src_words + tgt_words + event_words)
        event_set.update(get_events(event_line))
        stats["avg. input words"] += len(src_words)
        stats["avg. output words"] += len(tgt_words)
        stats["avg. events"] += len(event_words)
    stats["vocab"] = len(vocab)
    stats["events set"] = len(event_set)
    stats["avg. input words"] /= stats["stories"]
    stats["avg. output words"] /= stats["stories"]
    stats["avg. events"] /= stats["stories"]
    return  stats

def print_stats():
    for dataset_name in ["roc-stories", "writing-prompts"]:
        data_dir = Path(f"{BASE_DIR}/datasets/event-trigger/{dataset_name}")
        print(f"============== {dataset_name} ================")
        for prefix in ["train", "val", "test"]:
            stat_ = parse_stat(src_file=data_dir.joinpath(f"{prefix}.source.txt"),
                               tgt_file=data_dir.joinpath(f"{prefix}.target.txt"),
                               event_file=data_dir.joinpath(f"{prefix}_event.source.txt"))
            print(f"{prefix}: {stat_}")


if __name__ == '__main__':
    from preprocessing.event_trigger.event_extractor import EventExtractor  # cannot be removed

    print_stats()
