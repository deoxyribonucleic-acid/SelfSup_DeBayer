import os
import subprocess
import zipfile

def download_and_extract_sidd_medium(output_dir='SIDD_Medium'):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 定义所有分卷文件的下载链接
    base_url = 'https://competitions.codalab.org/my/datasets/download/'
    part_ids = [
        '6bf5aadd-ab08-43e2-bbff-f56b631b8d15',  # Part 0
        'f3fd938d-d693-44a2-bcda-e1ba4b9f60bc',  # Part 1
        '919a86de-25c7-4e62-84c8-ccccda67000d',  # Part 2
        '4e201cbc-0b50-428e-9dd6-8fce959883d3',  # Part 3
        '48d11e4b-67d1-4904-9562-265a457214a9',  # Part 4
        'c52e35aa-1e73-423f-a0a7-f95e13b8bbb9',  # Part 5
        '979a70d5-1306-4d32-a33e-7e9c925c5952',  # Part 6
        '9b596aa1-8562-4972-9d70-7826db1106dd',  # Part 7
        'fcb4b939-0ff6-497b-83d3-347d815ac395',  # Part 8
        '9f7e5e57-b324-43fd-8b15-530b90ff99ad',  # Part 9
        'fa9c37ca-d2df-4e64-8afd-1c4d985f600b',  # Part 10
        '697e2f7d-adfa-4bad-9df2-4b18188f70e4',  # Part 11
        '9fd75bba-29ed-4ccf-a305-f003116d8966',  # Part 12
        '96dd024a-8317-4649-b1b1-46989506fb94',  # Part 13
        '5f416d82-8342-4eaa-991a-ac1793141880',  # Part 14
        '65ce66e9-5231-4545-b5e0-4966ae5a476b',  # Part 15
        '79b6e06a-fecc-4320-a313-d329809f0831',  # Part 16
        '29d18514-5464-43a4-bfae-bd5be1adefee',  # Part 17
        'df801a17-c355-49d1-82e6-df1056b762b5',  # Part 18
        'f62d6786-ed79-4e12-b8ba-eb53b0dc17d7',  # Part 19
        '4edc9a00-e009-4fc1-912e-00910dcc32ff'   # Part 20
    ]

    # 下载所有分卷文件
    for idx, part_id in enumerate(part_ids):
        if idx == 0:
            filename = 'SIDD_Medium_Srgb_Parts.zip'
        else:
            filename = f'SIDD_Medium_Srgb_Parts.z{idx:02d}'
        file_path = os.path.join(output_dir, filename)
        if os.path.exists(file_path):
            print(f'文件已存在，跳过下载: {file_path}')
            continue
        url = base_url + part_id
        print(f'正在下载: {url} -> {filename}')
        try:
            subprocess.run([
                'wget',
                '-c',
                '-O', file_path,
                url
            ], check=True)
            print(f'下载完成: {file_path}')
        except subprocess.CalledProcessError as e:
            print(f'下载失败: {url}，错误: {e}')
            return

    # 解压主 zip 文件
    zip_file_path = os.path.join(output_dir, 'SIDD_Medium_Srgb_Parts.zip')
    print('开始解压 zip 文件...')
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print('解压完成。')
    except zipfile.BadZipFile as e:
        print(f'解压失败，可能是 zip 文件不完整或损坏。错误: {e}')

if __name__ == '__main__':
    download_and_extract_sidd_medium()
