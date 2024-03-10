"""
@Desc:
@Reference:
@Notes:
"""
from typing import List

from huggingface_hub import hf_hub_url, snapshot_download


def download_specific_file(repository_id: str = "lysandre/arxiv-nlp",
                           filename: str = "config.json"):
    hf_hub_url(repo_id=repository_id, filename=filename)


def download_repository(repository_id: str = "lysandre/arxiv-nlp",
                        ignore_regex: List[str] = ["*.msgpack", "*.h5", "*.tflite"],
                        resume_download=True):
    local_folder = snapshot_download(repo_id=repository_id,
                                     ignore_patterns=ignore_regex,
                                     resume_download=resume_download)
    print(f"{repository_id} downloaded in {local_folder}")


if __name__ == '__main__':
    download_repository(repository_id="THUDM/chatglm-6b")
