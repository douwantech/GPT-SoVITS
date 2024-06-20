# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, BaseModel, Input, Path, Secret
import subprocess
import os
import uuid
import requests
import shutil
from zipfile import ZipFile
import mimetypes
import re
import nltk
import oss2
import json

class Output(BaseModel):
    oss_zip_url: str
    zip_url: Path
    audio_url: Path

class Predictor(BasePredictor):
    def setup(self) -> None:
        nltk.download('cmudict')
        nltk.download('averaged_perceptron_tagger')

    def predict(
        self,
        audio_or_video_url: Path = Input(description="Train audio URL or video URL"),
        aliyun_oss_configure: Secret = Input(description='''If need upload to aliyun oss directly set this configure. 
            {
                "access_key_id": "your_access_key_id",
                "access_key_secret": "your_access_key_secret",
                "bucket_name": "your_bucket_name",
                "endpoint": "your_endpoint",
                "domain": "your_domain"
            }
        ''', default="")
    ) -> Output:
            real_uuid = str(uuid.uuid4())
            input_dir = f'input/{real_uuid}'
            os.makedirs(input_dir, exist_ok=True)
            input_file = f'{input_dir}/origin.mp3'
            log_file = f'{input_dir}/log.txt'
            with open(log_file, 'w') as file:
                pass  
            self.download_file(str(audio_or_video_url), input_file)

            mime_type, _ = mimetypes.guess_type(input_file)
            if mime_type and mime_type.startswith('video'):
                video_file = input_file
                input_file = f'{input_dir}/extracted_audio.wav'
                subprocess.run(['ffmpeg', '-i', video_file, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', input_file, '-y'], check=True)

            self.run_commands([
                f"python tools/slice_audio.py {input_file} output/{real_uuid}/slicer_opt -34 4000 100 10 500 0.9 0.25 0 1",
                f"python tools/cmd-denoise.py -i 'output/{real_uuid}/slicer_opt' -o 'output/{real_uuid}/denoise_opt' -p float16",
                f"python tools/asr/funasr_asr.py -i 'output/{real_uuid}/denoise_opt' -o 'output/{real_uuid}/asr_opt' -s large -l zh -p float16",
                f"python tools/prepare_data.py 'output/{real_uuid}/asr_opt/denoise_opt.list' 'output/{real_uuid}/denoise_opt' '{real_uuid}' '0-0' '0-0' '0-0' \
                    'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large' 'GPT_SoVITS/pretrained_models/chinese-hubert-base' 'GPT_SoVITS/pretrained_models/s2G488k.pth'"
            ], log_file)

            if self.is_previous_step_success(log_file, "一键三连进程结束"):
                self.run_commands([
                    f"python tools/train_sovits.py 11 8 '{real_uuid}' 0.4 1 1 4 '0-1' 'GPT_SoVITS/pretrained_models/s2G488k.pth' 'GPT_SoVITS/pretrained_models/s2D488k.pth'"
                ], log_file)
            else:
                raise BaseException("Prepare data failure")

            if self.is_previous_step_success(log_file, "SoVITS训练完成"):
                self.run_commands([
                    f"python tools/train_gpt.py 11 15 '{real_uuid}' 0 1 1 5 '0-1' 'GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt'"
                ], log_file)
            else:
                raise BaseException("Train Sovits Failure")

            if not self.is_previous_step_success(log_file, "GPT训练完成"):
                raise BaseException("Train GPT Failure")

            zip_path = self.zip_files(real_uuid, log_file, input_file)
            oss_zip_url = self.upload_file(aliyun_oss_configure, zip_path, "workers/zip")
            print(f"oss_zip_url: {oss_zip_url}")
            return Output(oss_zip_url=oss_zip_url, zip_url=Path(zip_path), audio_url=Path(input_file))

    def is_previous_step_success(self, log_file, keyword):
        try:
            with open(log_file, 'r') as file:
                log_contents = file.read()
                if keyword in log_contents:
                    return True
        except FileNotFoundError:
            print(f"Log file {log_file} not found.")
        except Exception as e:
            print(f"An error occurred while reading the log file: {e}")
        return False

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

    def run_commands(self, commands, log_file):
        for command in commands:
            try:
                self.execute_command(command, log_file)
            except subprocess.CalledProcessError as e:
                with open(log_file, 'a') as log:
                    log.write(f"Command '{e.cmd}' failed with return code {e.returncode}\n")
                print(f"Command '{e.cmd}' failed with return code {e.returncode}")

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

    def zip_files(self, real_uuid, log_file, input_file):
        if not os.path.exists("results"):
            os.makedirs("results")
        zip_filename = f'results/{real_uuid}.zip'
        with ZipFile(zip_filename, 'w') as zipf:
            # 添加原始输入文件
            zipf.write(input_file, "input.wav")

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
            pth_file = self.find_max_numbered_file('SoVITS_weights', real_uuid, ".pth")
            if pth_file is None:
                raise("Can't find pth file")
            zipf.write(pth_file, os.path.basename(pth_file))

            # 添加 GPT_SoVITS 目录下以 UUID 开头的文件
            ckpt_file = self.find_max_numbered_file('GPT_weights', real_uuid, ".ckpt")
            if ckpt_file is None:
                raise("Can't find ckpt file")
            zipf.write(ckpt_file, os.path.basename(ckpt_file))
        
        # 解压 ZIP 文件并列出内容
        with ZipFile(zip_filename, 'r') as zipf:
            zipf.extractall(f'results/{real_uuid}_extracted')
            print("Files in the zip archive:")
            for file in zipf.namelist():
                print(file)

        print(f"Created zip file: {zip_filename}")
        return zip_filename

    def find_max_numbered_file(self, directory, prefix, suffix):
        max_numbered_file = None
        max_number = -1

        for root, _, files in os.walk(directory):
            for file in files:
                if file.startswith(prefix) and file.endswith(suffix):
                    match = re.search(r'(\d+)' + re.escape(suffix), file)
                    if match:
                        number = int(match.group(1))
                        if number > max_number:
                            max_number = number
                            max_numbered_file = os.path.join(root, file)

        return max_numbered_file

    def upload_file(self, aliyun_oss_configure, file_path, target_dir):
        if isinstance(aliyun_oss_configure, Secret):
            aliyun_oss_configure = json.loads(aliyun_oss_configure.get_secret_value())
        else:
            return ""
        
        access_key_id = aliyun_oss_configure['access_key_id']
        access_key_secret = aliyun_oss_configure['access_key_secret']
        bucket_name = aliyun_oss_configure['bucket_name']
        endpoint = aliyun_oss_configure['endpoint']   
        domain = aliyun_oss_configure['domain']   

        auth = oss2.Auth(access_key_id, access_key_secret)
        bucket = oss2.Bucket(auth, endpoint, bucket_name)

        file_name = os.path.basename(file_path)
        target_path = f"{target_dir}/{file_name}"
        try:
            bucket.put_object_from_file(target_path, file_path)
            print(f"Successfully uploaded {file_name} to {target_path}")
            file_url = f"https://{domain}/{target_path}"
            return file_url
        except oss2.exceptions.OssError as e:
            raise BaseException(f"Failed to upload {file_name} to OSS: {e}")
            return None

        auth = oss2.Auth(access_key_id, access_key_secret)
        bucket = oss2.Bucket(auth, endpoint, bucket_name)

        # 获取文件名
        file_name = os.path.basename(file_path)
        target_path = f"{target_dir}/{file_name}"
        
        try:
            bucket.put_object_from_file(target_path, file_path)
            print(f"Successfully uploaded {file_name} to {target_path}")
            file_url = f"https://{bucket_name}.{endpoint}/{target_path}"
            return file_url
        except oss2.exceptions.OssError as e:
            raise BaseException(f"Failed to upload {file_name} to OSS: {e}")
            return None

def test():
    p = Predictor()
    p.setup()
    p.predict(audio_or_video_url="https://general-api.oss-cn-hangzhou.aliyuncs.com/static/2.mp4", aliyun_oss_configure=Secret('''
        {
            "access_key_id": "",
            "access_key_secret": "",
            "bucket_name": "",
            "endpoint": "",
            "domain": ""
        }
    '''))

#test()