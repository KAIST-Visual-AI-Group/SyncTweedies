import argparse 

def load_panorama_config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--app", type=str, default="panorama")
    
    # # Diffusion Config
    parser.add_argument('--prompt', type=str, default="a bedroom with a window")
    parser.add_argument('--negative_prompt', type=str, default='oversmoothed, blurry, depth of field, out of focus, low quality, bloom, glowing effect.')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--guidance_scale', type=float, default=25, help='Recommend above 12 to avoid blurriness')
    parser.add_argument('--seed', type=int, default=0)

    # ControlNet Config
    parser.add_argument('--cond_type', type=str, default='depth', help='Support depth and normal, less multi-face in normal mode, but some times less details')
    parser.add_argument('--conditioning_scale', type=float, default=0.7)
    parser.add_argument('--conditioning_scale_end', type=float, default=0.9, help='Gradually increasing conditioning scale for better geometry alignment near the end')
    parser.add_argument('--control_guidance_start', type=float, default=0.0)
    parser.add_argument('--control_guidance_end', type=float, default=0.99)

    # Multi-View Config
    parser.add_argument('--mvd_end', type=float, default=0.8, help='Time step to stop texture space aggregation')
    parser.add_argument('--ref_attention_end', type=float, default=0.2, help='Lower->better quality; higher->better harmonization')
    parser.add_argument("--case_num", type=str, default="2", required=True) # texture image

    # Logging Config
    parser.add_argument('--log_interval', type=int, default=5) # 10 -> 5
    parser.add_argument('--max_batch_size', type=int, default=48)
    
    # for panorama360
    parser.add_argument("--average_rgb", action="store_true", default=False)
    parser.add_argument("--canonical_rgb_h", type=int, default=1024)
    parser.add_argument("--canonical_rgb_w", type=int, default=2048)
    parser.add_argument("--canonical_latent_h", type=int, default=2048)
    parser.add_argument("--canonical_latent_w", type=int, default=4096)
    parser.add_argument("--instance_latent_size", type=int, default=64)
    parser.add_argument("--instance_rgb_size", type=int, default=512)
    parser.add_argument("--theta_range", nargs=2, type=int, default=[0,360])
    parser.add_argument("--theta_interval", type=int, default=45)
    parser.add_argument("--phi_range", nargs=2, type=int, default=[0, 1])
    parser.add_argument("--phi_interval", type=int, default=45)
    parser.add_argument("--FOV", type=int, default=72)
    parser.add_argument("--voronoi", action="store_true", default=True) 

    parser.add_argument("--model", type=str, default="controlnet") # controlnet / sd / deepfloyd
    parser.add_argument("--depth_data_path", type=str, help="/PATH/TO/file.dpt")

    ## logging
    parser.add_argument("--save_top_dir", type=str, default="results/panorama360_depth")
    parser.add_argument("--tag", type=str)
    parser.add_argument("--save_dir_now", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--initialize_xt_from_zt", action="store_true")
    parser.add_argument("--sampling_method", type=str, default="ddim")
    
    options = parser.parse_args()
    
    return options
