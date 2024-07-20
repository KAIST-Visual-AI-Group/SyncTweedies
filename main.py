import sys 
import os 
import numpy as np 
import torch


def seed_everything(seed=2024):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    app = sys.argv[sys.argv.index("--app")+1]
    seed_everything()

    if app == "ambiguous_image":
        from synctweedies.model.ambiguous_image_model import AmbiguousImageModel
        from synctweedies.config.ambiguous_image_config import load_ambiguious_image_config

        config = load_ambiguious_image_config()
        model = AmbiguousImageModel(config)
        
    elif app == "wide_image":
        from synctweedies.model.wide_image_model import WideImageModel
        from synctweedies.config.wide_image_config import load_wide_image_config

        config = load_wide_image_config()
        model = WideImageModel(config)

    elif app == "panorama":
        from synctweedies.model.panorama_model import PanoramaModel
        from synctweedies.config.panorama_config import load_panorama_config

        config = load_panorama_config()
        model = PanoramaModel(config)

    elif app == "mesh":
        from synctweedies.model.mesh_texture_model import MeshTextureModel
        from synctweedies.config.mesh_config import load_mesh_config

        config = load_mesh_config()
        model = MeshTextureModel(config)

    elif app == "gs":
        from synctweedies.model.gaussian_splatting_model import GaussianSplattingModel
        from synctweedies.config.gs_config import load_gs_config

        config = load_gs_config()
        model = GaussianSplattingModel(config)

    else:
        raise NotImplementedError(f"{app} not implemented")

    model()
    
