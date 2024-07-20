import argparse 

def load_ambiguious_image_config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--app", type=str, default="ambiguous_image")
    parser.add_argument("--sampling_method", type=str, default="ddim")
    parser.add_argument("--prompts", nargs="*", default=["a horse", "a snowy mountain village"])

    parser.add_argument("--stage_1_path", type=str, default="DeepFloyd/IF-I-M-v1.0")
    parser.add_argument("--stage_2_path", type=str, default="DeepFloyd/IF-II-M-v1.0")
    parser.add_argument("--model", type=str, default="deepfloyd")
    parser.add_argument("--sd_model_name", type=str, default="deepfloyd")

    parser.add_argument("--style", type=str, default="an oilpainting of")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=10)
    parser.add_argument("--case_num", type=str, default="2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stage2_noise_level", type=int, default=50)
    parser.add_argument("--views_names", nargs="*", default=["identity", "rotate_cw"])
    parser.add_argument("--rotate_angle", type=int, default=45)
    
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--initialize_xt_from_zt", action="store_true")
    parser.add_argument("--initialize_same_xt", action="store_true")
    parser.add_argument("--scaling_factor", type=float, default=1.0)
    
    # Logging
    parser.add_argument('--save_top_dir', type=str, default="./output")
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--save_dir_now', action='store_true')
    parser.add_argument("--log_step", type=int, default=10)

    # for n-1 Projection
    parser.add_argument("--optimize_inverse_mapping", action="store_true")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num_of_zs", type=int, default=10)

    options = parser.parse_args()

    return options

    
    

    