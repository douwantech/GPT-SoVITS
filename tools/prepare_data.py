import os,sys,numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from webui import open1abc

def add_prefix_to_file_in_place(file_path, prefix):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(file_path, 'w', encoding='utf-8') as file:
        for line in lines:
            if line.startswith('output/'):
                modified_line = line.replace('output/', f'{prefix}/output/', 1)
            else:
                modified_line = line
            file.write(modified_line)

#add_prefix_to_file_in_place(sys.argv[1], '/src')
generator = open1abc(*sys.argv[1:])
for result in generator:
    print(result)
