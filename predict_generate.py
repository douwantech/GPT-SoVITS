# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import subprocess
import os
import uuid
import requests
import shutil
from zipfile import ZipFile
from funasr import AutoModel
from tqdm import tqdm

class Predictor(BasePredictor):
    def __init__(self):
        path_asr  = 'tools/asr/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
        path_vad  = 'tools/asr/models/speech_fsmn_vad_zh-cn-16k-common-pytorch'
        path_punc = 'tools/asr/models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'
        path_asr  = path_asr  if os.path.exists(path_asr)  else "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        path_vad  = path_vad  if os.path.exists(path_vad)  else "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
        path_punc = path_punc if os.path.exists(path_punc) else "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
        self.asr_model = AutoModel(
            model               = path_asr,
            model_revision      = "v2.0.4",
            vad_model           = path_vad,
            vad_model_revision  = "v2.0.4",
            punc_model          = path_punc,
            punc_model_revision = "v2.0.4",
        )

    def setup(self) -> None:
        return

    def predict(
        self,
        zip_url: Path = Input(description="Train zip package"),
        text: str = Input(description="Inference text")
    ) -> Path:
            log_file = 'execution_log.txt'
            input_file = f'train.zip'
            output_dir = 'unzipped'
            self.download_file(str(zip_url), input_file)
            self.unzip_file(input_file, output_dir)
            sovites_model_path = self.find_file(output_dir, '.pth')
            gpt_model_path = self.find_file(output_dir, '.ckpt')
            ref_wav = output_dir + "/denoise_opt/" + self.get_ref_wav(os.path.join(output_dir, 'asr_opt/denoise_opt.list'))

            commands = [
                f"ffmpeg -i {ref_wav} -t 3 -y ref.wav",
            ]
            ref_text = self.asr_model.generate(input="ref.wav")[0]["text"]
            print(f"Ref text: {ref_text}")

            commands = [
                f"python tools/generate.py {sovites_model_path} {gpt_model_path} ref.wav {ref_text} {text}",
            ]

            if os.path.exists(log_file):
                os.remove(log_file)  

            for command in commands:
                try:
                    self.execute_command(command, log_file)
                except subprocess.CalledProcessError as e:
                    with open(log_file, 'a') as log:
                        log.write(f"Command '{e.cmd}' failed with return code {e.returncode}\n")
                    print(f"Command '{e.cmd}' failed with return code {e.returncode}")
        
            return Path("result.wav")


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
                print(stdout_line, end='')  
                log.write(stdout_line)  
            process.stdout.close()
            return_code = process.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, command)

    def unzip_file(self, zip_path, dest_dir):
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        print(f"Extracted {zip_path} to {dest_dir}")

    def find_file(self, directory, extension):
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(extension):
                    return os.path.join(root, file)
        raise FileNotFoundError(f"No file with extension {extension} found in {directory}")

    def get_ref_wav(self, list_path):
        with open(list_path, 'r') as file:
            first_line = file.readline().strip()
            ref_wav_path = first_line.split('|')[0]
            return os.path.basename(ref_wav_path)