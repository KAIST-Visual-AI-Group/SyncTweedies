import subprocess

command = f'python main.py \
    --app ambiguous_image \
    --case_num 2 \
    --tag ambiguous_image \
    --save_dir_now' 
    
subprocess.call(command, shell=True)
print("Done")