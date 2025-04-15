import os
import requests
from tqdm import tqdm

# 创建保存路径
save_dir = './SIDD_Validation_Blocks'
os.makedirs(save_dir, exist_ok=True)

# 文件名和 Mirror 1 URL 映射
files = {
    "ValidationNoisyBlocksRaw.mat": "https://competitions.codalab.org/my/datasets/download/549374a5-5e7d-4daf-a977-993187b2d050",
    "ValidationNoisyBlocksSrgb.mat": "https://competitions.codalab.org/my/datasets/download/260ed944-21e7-4317-bd66-a6dd42343838",
    "ValidationGtBlocksRaw.mat": "https://competitions.codalab.org/my/datasets/download/6d90a38a-6d6d-4dff-a3ab-413f5208b7bd",
    "ValidationGtBlocksSrgb.mat": "https://competitions.codalab.org/my/datasets/download/5349efe0-dfa2-499e-a46f-49c623285ba7",
}

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(save_path, 'wb') as file, tqdm(
        desc=os.path.basename(save_path),
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))

# 执行下载
for filename, url in files.items():
    save_path = os.path.join(save_dir, filename)
    if os.path.exists(save_path):
        print(f"[✓] 已存在: {filename}, 跳过下载")
    else:
        print(f"[↓] 正在下载: {filename}")
        try:
            download_file(url, save_path)
        except Exception as e:
            print(f"[X] 下载失败: {filename}, 错误: {e}")
