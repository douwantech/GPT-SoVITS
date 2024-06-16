import os,sys,numpy as np
import scipy.io.wavfile as wavfile

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir + "/GPT_SoVITS")
from inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

def str_to_bool(s):
    if s == "1":
        return True
    elif s == "0":
        return False
    else:
        raise ValueError(f"Invalid input: {s}")

change_sovits_weights(sys.argv[1])
change_gpt_weights(sys.argv[2])
result = get_tts_wav(sys.argv[3], sys.argv[4], "Chinese", sys.argv[5], "Chinese")
sampling_rate, audio_data = next(result)
wavfile.write(sys.argv[6], sampling_rate, audio_data)
