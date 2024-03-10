import os
import csv
import json
from tqdm import tqdm
from pathlib import Path
import sys
import glob
import os
import re
from pathlib import Path
import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas
import shutil

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from src.utils.file_utils import load_json, load_jsonl, lines_to_file


def find_gen_file(input_path: Path):
    gen_file_path = None
    # 后缀为 _gen.txt 的文件名模式
    pattern = r'^.*_gen\.txt$'

    # 遍历目录
    for root, dirs, files in os.walk(input_path):
        for file in files:
            # 使用正则表达式匹配文件名
            if re.match(pattern, file):
                # 构建文件的完整路径
                file_path = os.path.join(root, file)
                # 添加到匹配的文件路径列表
                gen_file_path = Path(file_path)
    return gen_file_path


def collect_files_for_exp(dataset_dir: Path, output_dir: Path):
    dest_dir = output_dir / "gen_files_for_exp"
    dest_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in ["roc-stories", "writing-prompts"]:

        # copy datasets

        sp_dataset_dir = dataset_dir / dataset_name
        # lead_file = sp_dataset_dir / "test.source.txt"
        # event_file = sp_dataset_dir / "test_event.source.txt"
        tgt_file = sp_dataset_dir / "test.target.txt"
        # shutil.copy(lead_file, dest_dir / f"{dataset_name}-test.source.txt")
        # shutil.copy(event_file, dest_dir / f"{dataset_name}-test_event.source.txt")
        shutil.copy(tgt_file, dest_dir / f"{dataset_name}-golden_gen.txt")

if __name__ == '__main__':
    dataset_dir = Path(f"{BASE_DIR}/datasets/event-trigger")
    output_dir = Path(f"{BASE_DIR}/output/cetrica/")
    collect_files_for_exp(dataset_dir, output_dir)



