"""下载固定版本的 PersonaMem CSV 和它实际引用的 32k 历史。"""

import csv
import hashlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import HTTPException
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen


REPOSITORY = "bowen-upenn/PersonaMem-v2"
REVISION = "ed956dea41521fc4499acbc63f966e0fd3c053ba"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "data" / "personamem-v2"


def download(remote_path: str, local_path: str) -> tuple[str, str]:
    destination = (OUTPUT / local_path).resolve()
    if not destination.is_relative_to(OUTPUT.resolve()):
        raise ValueError(f"dataset path leaves output directory: {local_path}")
    # JSON 是仓库中的文本文件，直接读取 raw；CSV 使用 resolve 获取完整数据。
    endpoint = "raw" if remote_path.endswith(".json") else "resolve"
    url = f"https://huggingface.co/datasets/{REPOSITORY}/{endpoint}/{REVISION}/{quote(remote_path, safe='/')}"
    for attempt in range(3):
        try:
            with urlopen(url, timeout=60) as response:
                content = response.read()
            break
        except (HTTPException, OSError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    # 允许失败后重跑，但不覆盖与固定数据版本不一致的已有文件。
    if destination.exists():
        if destination.read_bytes() != content:
            raise ValueError(f"existing dataset file differs from pinned source: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    return local_path, hashlib.sha256(content).hexdigest()


def main():
    csv_path, csv_hash = download("benchmark/text/benchmark.csv", "benchmark.csv")
    content = (OUTPUT / csv_path).read_bytes()
    reference = PROJECT_ROOT / "data" / "personamem" / "benchmark.csv"
    if reference.exists() and reference.read_bytes() != content:
        raise ValueError("pinned PersonaMem CSV differs from the existing benchmark")
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"))))
    histories = sorted({row["chat_history_32k_link"] for row in rows})
    files = {csv_path: csv_hash}
    print(f"Preparing {len(rows)} questions and {len(histories)} histories at {REVISION}", flush=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        pending = [executor.submit(download, path, path) for path in histories]
        for index, future in enumerate(as_completed(pending), 1):
            path, digest = future.result()
            files[path] = digest
            if index % 25 == 0 or index == len(histories):
                print(f"Downloaded {index}/{len(histories)} histories", flush=True)
    source = {"repository": REPOSITORY, "revision": REVISION,
              "benchmark_path": "benchmark/text/benchmark.csv", "question_count": len(rows),
              "history_count": len(histories), "files": dict(sorted(files.items()))}
    (OUTPUT / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Dataset ready: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
