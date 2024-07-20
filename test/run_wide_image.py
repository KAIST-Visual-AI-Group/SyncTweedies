import subprocess

command = f'python main.py \
    --app wide_image \
    --prompt "A photo of a mountain range at twilight" \
    --save_top_dir ./output \
    --case_num 2 \
    --seed 0 \
    --sampling_method ddim \
    --num_inference_steps 50 \
    --panorama_height 512 \
    --panorama_width 3072 \
    --mvd_end 1.0 \
    --initialize_xt_from_zt \
    --save_dir_now \
    --tag wide_image'
    
subprocess.call(command, shell=True)
print("Done")