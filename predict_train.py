# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import subprocess
import os
import uuid
import requests
import shutil
from zipfile import ZipFile

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
        audio_url: Path = Input(description="Human audio url")
    ) -> Path:
            log_file = 'execution_log.txt'
            real_uuid = str(uuid.uuid4())
            input_dir = f'input/{real_uuid}'
            os.makedirs(input_dir, exist_ok=True)
            input_file = f'{input_dir}/origin.mp3'

            # URL of the file to download
            self.download_file(str(audio_url), input_file)

            commands = [
                f"python tools/slice_audio.py {input_file} output/{real_uuid}/slicer_opt -34 4000 100 10 500 0.9 0.25 0 1",
                f"python tools/cmd-denoise.py -i 'output/{real_uuid}/slicer_opt' -o 'output/{real_uuid}/denoise_opt' -p float16",
                f"python tools/asr/funasr_asr.py -i 'output/{real_uuid}/denoise_opt' -o 'output/{real_uuid}/asr_opt' -s large -l zh -p float16",
                f"python tools/prepare_data.py 'output/{real_uuid}/asr_opt/denoise_opt.list' 'output/{real_uuid}/denoise_opt' '{real_uuid}' '0-0' '0-0' '0-0' \
                    'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large' 'GPT_SoVITS/pretrained_models/chinese-hubert-base' 'GPT_SoVITS/pretrained_models/s2G488k.pth'",
                f"python tools/train_sovits.py 11 8 '{real_uuid}' 0.4 1 1 4 '0-1' 'GPT_SoVITS/pretrained_models/s2G488k.pth' 'GPT_SoVITS/pretrained_models/s2D488k.pth'",
                f"python tools/train_gpt.py 11 15 '{real_uuid}' 0 1 1 5 '0-1' 'GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt'"
            ]

            if os.path.exists(log_file):
                os.remove(log_file)  # 如果日志文件已存在，则删除它以确保新的开始

            for command in commands:
                try:
                    self.execute_command(command, log_file)
                except subprocess.CalledProcessError as e:
                    with open(log_file, 'a') as log:
                        log.write(f"Command '{e.cmd}' failed with return code {e.returncode}\n")
                    print(f"Command '{e.cmd}' failed with return code {e.returncode}")
        
            zip_path = self.zip_files(real_uuid, log_file)
            return Path(zip_path)


    def download_file(self, url, dest_path):
        if url.startswith('http://') or url.startswith('https://'):
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(dest_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
            print(f"Downloaded file to {dest_path}")
        else:
            shutil.copy(url, dest_path)
            print(f"Copied file to {dest_path}")

    def execute_command(self, command, log_file):
        with open(log_file, 'a') as log:
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            for stdout_line in iter(process.stdout.readline, ""):
                print(stdout_line, end='')  # 打印到控制台
                log.write(stdout_line)  # 写入日志文件
            process.stdout.close()
            return_code = process.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, command)

    def zip_files(self, real_uuid, log_file):
        zip_filename = f'{real_uuid}.zip'
        with ZipFile(zip_filename, 'w') as zipf:
            # 添加日志文件
            zipf.write(log_file, os.path.basename(log_file))
            
            # 添加 denoise_opt.list 文件
            denoise_list_file = f'output/{real_uuid}/asr_opt/denoise_opt.list'
            zipf.write(denoise_list_file, os.path.relpath(denoise_list_file, f'output/{real_uuid}'))

            # 添加 denoise_opt 目录下的所有文件
            denoise_opt_dir = f'output/{real_uuid}/denoise_opt'
            for root, _, files in os.walk(denoise_opt_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, f'output/{real_uuid}'))
            
            # 添加 SoVITS_weights 目录下以 UUID 开头的文件
            sovits_weights_dir = 'SoVITS_weights'
            for root, _, files in os.walk(sovits_weights_dir):
                for file in files:
                    if file.startswith(real_uuid):
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, sovits_weights_dir))

            # 添加 GPT_SoVITS 目录下以 UUID 开头的文件
            gpt_sovits_dir = 'GPT_SoVITS'
            for root, _, files in os.walk(gpt_sovits_dir):
                for file in files:
                    if file.startswith(real_uuid):
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, gpt_sovits_dir))
        
        print(f"Created zip file: {zip_filename}")
        return zip_filename