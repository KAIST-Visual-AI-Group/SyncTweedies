import subprocess

command = f'python main.py \
    --app mesh \
    --prompt "A hand carved wood turtle" \
    --save_top_dir ./output \
    --tag mesh \
    --case_num 2 \
    --mesh ./data/mesh/turtle.obj \
    --steps 30 \
    --seed 0 \
    --sampling_method ddim \
    --initialize_xt_from_zt \
    --guidance_scale 15.5 \
    --save_dir_now'    
    
subprocess.call(command, shell=True)
print("Done")