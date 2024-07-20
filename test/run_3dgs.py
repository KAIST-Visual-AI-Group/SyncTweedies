import subprocess

command = f'python main.py \
    --app gs \
    --tag gs \
    --save_dir_now \
    --save_top_dir ./output \
    --prompt "A photo of majestic red throne, adorned with gold accents" \
    --source_path ./data/gaussians/chair \
    --plyfile ./data/gaussians/chair.ply \
    --dataset_type blender \
    --case_num 2 \
    --zt_init \
    --force_clean_composition'
    
subprocess.call(command, shell=True)
print("Done")
