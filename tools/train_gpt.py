import os,sys,numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from webui import open1Bb

def str_to_bool(s):
    if s == "1":
        return True
    elif s == "0":
        return False
    else:
        raise ValueError(f"Invalid input: {s}")

generator = open1Bb(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], str_to_bool(sys.argv[4]), str_to_bool(sys.argv[5]), str_to_bool(sys.argv[6]), int(sys.argv[7]), sys.argv[8], sys.argv[9])
for result in generator:
    print(result)