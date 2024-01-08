import requests
from pathlib import Path
from tqdm import tqdm

# refo : https://github.com/hkproj/pytorch-stable-diffusion/blob/main/sd/model_converter.py
# https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main
# pip install requests

# 다운로드할 파일들의 URL 리스트
file_urls = [
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt?download=true",
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/vocab.json?download=true",
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/merges.txt?download=true"
]

# 저장할 경로 지정
#save_path = Path("D:/python/CodeIntelligence/StableDiffusion/data")
save_path = Path("/Users/a24/Desktop/pyskillup/CodeIntelligence/stablediffusion/StableDiffusion/data")


# 해당 경로가 없으면 생성
save_path.mkdir(parents=True, exist_ok=True)

# 파일을 다운로드하고 지정된 경로에 저장하는 함수


def download_file(url):
    local_filename = save_path / Path(url.split("?")[0]).name
    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # 여기에 추가됨. HTTP 요청이 실패하면 예외를 발생시킵니다.

        total_size_in_bytes = int(r.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte

        progress_bar = tqdm(total=total_size_in_bytes,
                            unit='iB', unit_scale=True)
        with open(local_filename, 'wb') as file:
            for chunk in r.iter_content(block_size):
                progress_bar.update(len(chunk))
                file.write(chunk)
        progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
    return local_filename


# 모든 파일을 다운로드
for url in file_urls:
    filename = download_file(url)
    print(f'Downloaded {filename}')
