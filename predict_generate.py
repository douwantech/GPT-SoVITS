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
        real_uuid = str(uuid.uuid4())
        output_dir = f'results/{real_uuid}'
        os.makedirs(output_dir, exist_ok=True)

        log_file = os.path.join(output_dir, 'execution_log.txt')
        input_file = os.path.join(output_dir, 'train.zip')
        unzip_dir = os.path.join(output_dir, 'unzipped')

        self.download_file(str(zip_url), input_file)
        self.unzip_file(input_file, unzip_dir)
        sovites_model_path = self.find_file(unzip_dir, '.pth')
        gpt_model_path = self.find_file(unzip_dir, '.ckpt')
        ref_name = self.get_ref_wav(os.path.join(unzip_dir, 'asr_opt/denoise_opt.list'))
        ref_wav = os.path.join(unzip_dir, "denoise_opt", ref_name)
        split_ref_wav = os.path.join(output_dir, 'ref.wav')

        self.run_commands([
            f"ffmpeg -i {ref_wav} -t 3 -y {split_ref_wav}",
        ], log_file)
        ref_text = self.asr_model.generate(input=split_ref_wav)[0]["text"]
        print(f"Ref text: {ref_text}")

        result_path = os.path.join(output_dir, "result.wav")
        self.run_commands([
            f"python tools/generate.py '{sovites_model_path}' '{gpt_model_path}' '{split_ref_wav}' '{ref_text}' '{text}' '{result_path}'",
        ], log_file)

        return Path(result_path)

    def run_commands(self, commands, log_file):
        for command in commands:
            try:
                self.execute_command(command, log_file)
            except subprocess.CalledProcessError as e:
                with open(log_file, 'a') as log:
                    log.write(f"Command '{e.cmd}' failed with return code {e.returncode}\n")
                print(f"Command '{e.cmd}' failed with return code {e.returncode}")

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
def test():
    # 示例用法
    p = Predictor()
    p.predict(
        zip_url="https://replicate.delivery/pbxt/uEMFe94O1W2dUSts7QdLtqrQZfSXw3je6LKfGNjcir0kzf3XC/89b88961-bc69-4789-9c96-4bc7866a4dff.zip",
        text="我该跟你说些什么好呢,人生世事无常,及时行乐,不必太纠结当下的困境"
    )

#test()