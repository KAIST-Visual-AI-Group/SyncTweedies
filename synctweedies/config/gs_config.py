import argparse

def load_gs_config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--app", type=str, default="gs")
    
    # File Config
    parser.add_argument('--save_top_dir', type=str, required=True, default=None, help="If not provided, use the parent directory of config file for output")
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--save_dir_now', action='store_true')
    parser.add_argument('--log_interval', type=int, default=2) # 10 -> 5

    # Diffusion Config
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--negative_prompt', type=str, default='oversmoothed, blurry, depth of field, out of focus, low quality, bloom, glowing effect.')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--guidance_scale', type=float, default=35, help='Recommend above 12 to avoid blurriness')
    parser.add_argument("--sampling_method", type=str, default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)

    # ControlNet
    parser.add_argument('--model', type=str, default='controlnet')
    parser.add_argument('--cond_type', type=str, default='depth', help='Support depth and normal, less multi-face in normal mode, but some times less details')
    parser.add_argument('--guess_mode', action='store_true')
    parser.add_argument('--conditioning_scale', type=float, default=0.7)
    parser.add_argument('--conditioning_scale_end', type=float, default=0.9, help='Gradually increasing conditioning scale for better geometry alignment near the end')
    parser.add_argument('--control_guidance_start', type=float, default=0.0)
    parser.add_argument('--control_guidance_end', type=float, default=0.99)
    parser.add_argument('--guidance_rescale', type=float, default=0.0, help='Not tested')

    # Mapper & Renderer settings
    parser.add_argument("--force_clean_composition", action="store_true", default=False)
    parser.add_argument("--dataset_type", type=str, default="colmap", choices=["dense", "sparse", "colmap", "blender"])
    parser.add_argument("--plyfile", type=str, default=None)
    parser.add_argument('--source_path', type=str, default=None, required=False, help="source_path must be provided for colmap and blender dataset")
    parser.add_argument("--up_vec", type=str, default="z", choices=["z", "y"])
    parser.add_argument('--antialiased', action='store_true')
    parser.add_argument('--no-antialiased', dest='antialiased', action='store_false')
    parser.set_defaults(antialiased=True)
    parser.add_argument("--zt_init", action="store_true", default=False)
    parser.add_argument("--background_color", type=str, default="white", choices=["white", "black"])
    parser.add_argument("--color_assign_method", type=str, default="previous", choices=["zero", "previous", "instance"])
    
    parser.add_argument("--canonical_domain", type=str, default="rgb", choices=["rgb", "latent"])

    # Multi-View Config
    parser.add_argument("--case_num", type=str, required=True) # texture image
    parser.add_argument("--total_image_to_use", default=50, type=int)
    parser.add_argument('--latent_view_size', type=int, default=64, help='Larger resolution, less aliasing in latent images; quality may degrade if much larger trained resolution of networks')
    parser.add_argument('--rgb_view_size', type=int, default=512)
    parser.add_argument('--mvd_end', type=float, default=0.8, help='Time step to stop texture space aggregation')
    parser.add_argument('--ref_attention_end', type=float, default=0.2, help='Lower->better quality; higher->better harmonization')
    parser.add_argument('--max_batch_size', type=int, default=50)

    # 3DGS Settings
    parser.add_argument('--iterations', default=1000, type=int)
    parser.add_argument('--final_iterations', default=4000, type=int)
    parser.add_argument('--xyz_lr', default=3e-4, type=float)
    parser.add_argument('--scale_lr', default=0.005, type=float)
    parser.add_argument('--quat_lr', default=0.001, type=float)
    parser.add_argument('--opacity_lr', default=0.05, type=float)
    parser.add_argument('--feature_lr', default=0.025, type=float)

    parser.add_argument('--enable_opacity', action="store_true", default=False)
    parser.add_argument('--enable_xyz', action="store_true", default=False)
    parser.add_argument('--enable_covariance', action="store_true", default=False)

    options = parser.parse_args()

    return options
