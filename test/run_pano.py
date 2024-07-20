import subprocess

command = f'python main.py \
    --app panorama \
    --tag panorama \
    --prompt "An old looking library" \
    --depth_data_path ./data/panorama/cf726b6c0144425282245b34fc4efdca_depth.dpt \
    --case_num 2 \
    --average_rgb \
    --model controlnet \
    --save_top_dir ./output \
    --save_dir_now'
subprocess.call(command, shell=True)
print("Done")