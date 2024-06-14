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
            ref_wav = "denoise_opt/" + self.get_ref_wav(os.path.join(output_dir, 'asr_opt/denoise_opt.list'))
            ref_text = "大家好，呃，最近发视频"

            commands = [
                f"ffmpeg -i {output_dir}/{ref_wav} -t 3 -y ref.wav",
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
