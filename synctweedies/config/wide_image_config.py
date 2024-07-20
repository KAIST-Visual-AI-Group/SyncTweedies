import argparse 

def load_wide_image_config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--app", type=str, default="wide_image")
    
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--negative_prompt', type=str, default="")
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=7.5)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument('--case_num', type=str, default="2")
    parser.add_argument('--sampling_method', type=str, default="ddim")
    parser.add_argument('--model', type=str, default="sd")
    parser.add_argument('--sd_path', type=str, default="stabilityai/stable-diffusion-2-base") # stabilityai/stable-diffusion-2-base / runwayml/stable-diffusion-v1-5
    parser.add_argument('--ref_attention_end', type=float, default=0.2)
    parser.add_argument('--mvd_end', type=float, default=0.8)
    
    # Logging 
    parser.add_argument("--save_top_dir", type=str, default="results/panorama360_depth")
    parser.add_argument("--tag", type=str)
    parser.add_argument("--save_dir_now", action="store_true")
    parser.add_argument('--log_step', type=int, default=10)
    
    # Resolution / Sizes
    parser.add_argument('--eval_w', type=int, action='append', default=[0, 10, 20, 30, 40, 50, 60, 70])
    parser.add_argument('--eval_h', type=int, action='append', default=[0, 0, 0, 0, 0, 0, 0, 0])
    parser.add_argument('--initialize_xt_from_zt', action='store_true')
    parser.add_argument('--window_stride', type=int, default=8)
    parser.add_argument('--latent_instance_size', type=int, default=64)
    parser.add_argument('--rgb_instance_size', type=int, default=512)
    parser.add_argument('--panorama_height', type=int, default=512)
    parser.add_argument('--panorama_width', type=int, default=3072)
    
    
    options = parser.parse_args()
    
    return options