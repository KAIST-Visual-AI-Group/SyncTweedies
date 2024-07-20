import numpy as np
import torch 
from PIL import Image
import cv2
import os
from tqdm import tqdm
from natsort import natsorted
import imageio

def pil_to_torch(pil_img):
    _np_img = np.array(pil_img).astype(np.float32) / 255.0
    _torch_img = torch.from_numpy(_np_img).permute(2, 0, 1).unsqueeze(0)
    return _torch_img

def torch_to_pil(tensor):
    if tensor.dim() == 4:
        _b, *_ = tensor.shape
        if _b == 1:
            tensor = tensor.squeeze(0)
        else:
            tensor = tensor[0, ...]
    
    tensor = tensor.permute(1, 2, 0)
    np_tensor = tensor.detach().cpu().numpy()
    np_tensor = (np_tensor * 255.0).astype(np.uint8)
    pil_tensor = Image.fromarray(np_tensor)
    return pil_tensor

def torch_to_pil_batch(tensor, is_grayscale=False, cmap=None):
    if is_grayscale:
        assert tensor.dim() == 2 or tensor.dim() == 3 or (tensor.dim() == 4 and tensor.shape[1] == 1)  # HW or BHW or B1HW
        # Make them all 4D tensor
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # 1HW
        elif tensor.dim() == 3:
            tensor = tensor.unsqueeze(1)  # B1HW
        
        # colormap
        if cmap is not None:
            raise NotImplementedError("Not implemented yet")
        else:
            tensor = tensor.repeat(1, 3, 1, 1)
    else:
        assert (tensor.dim() == 3 and tensor.shape[0] == 3) or (tensor.dim() == 4 and tensor.shape[1] == 3)  # 3HW or B3HW
        # Make them all 4D tensor
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
    
    tensor = tensor.clamp(0, 1)  # B3HW
    assert tensor.dim() == 4 and tensor.shape[1] == 3, f"Invalid tensor shape: {tensor.shape}"
    tensor = tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
    tensor = (tensor * 255.0).astype(np.uint8)
    pil_tensors = [Image.fromarray(image) for image in tensor]
    return pil_tensors

def pt_to_pil(images):
    """
    Convert a torch image to a PIL image.
    """
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    images = numpy_to_pil(images)
    return images


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def merge_images(images):
    if isinstance(images[0], Image.Image):
        return stack_images_horizontally(images)

    images = list(map(stack_images_horizontally, images))
    return stack_images_vertically(images)


def stack_images_horizontally(images, save_path=None):
    widths, heights = list(zip(*(i.size for i in images)))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new("RGBA", (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    if save_path is not None:
        new_im.save(save_path)
    return new_im


def stack_images_vertically(images, save_path=None):
    widths, heights = list(zip(*(i.size for i in images)))
    max_width = max(widths)
    total_height = sum(heights)
    new_im = Image.new("RGBA", (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    if save_path is not None:
        new_im.save(save_path)
    return new_im


def save_video(images, filename, fps=10):
    # Save PIL images as video
    writer = imageio.get_writer(filename, fps=fps)
    for img in images:
        img = np.array(img)
        writer.append_data(img)
    writer.close()

def save_gif(images, filename, fps=10):
    # Save PIL images as gif
    images[0].save(filename, save_all=True, append_images=images[1:], loop=0, duration=int(1000//fps))

def make_video(input_paths, output_path, fps=10, force=False):
    if os.path.exists(output_path) and not force:
        print(f"File {output_path} already exists. Skipping.")
        return
    writer = imageio.get_writer(output_path, fps=fps)
    for input_path in input_paths:
        img = imageio.imread(input_path)
        writer.append_data(img)
    writer.close()

def make_gif(input_path, output_path, fps=10, force=False):
    if os.path.exists(output_path) and not force:
        print(f"File {output_path} already exists. Skipping.")
        return
    # load video or images, depending on the input_path(list or str ends with .mp4)
    images = []
    if input_path.endswith('.mp4'):
        reader = imageio.get_reader(input_path)
        for img in reader:
            images.append(Image.fromarray(img))
    else:
        assert type(input_path) == list, "input_path should be a list of image paths"
        for img_path in input_path:
            img = imageio.imread(img_path)
            images.append(Image.fromarray(img))
    save_gif(images, output_path, fps=fps)


def make_video_recursive(input_dir, output_dir, fps=10):
    # Recursive search for 'image folder' make video, and save it to 'video folder'.mp4
    for root, dirs, files in tqdm(os.walk(input_dir)):
        files = [file for file in files if file.endswith(('.jpg', '.jpeg', '.png'))]  # filter only images files
        if len(files) >= 2:
            files = natsorted(files)  # sort files in natural order
            input_paths = [os.path.join(root, file) for file in files]
            if root == input_dir:
                orig_base = os.path.basename(root)
                target_dir = output_dir
                target_path = os.path.join(output_dir, orig_base + ".mp4")
            else:
                orig_dir, orig_base = os.path.split(root)
                target_dir = os.path.join(output_dir, os.path.relpath(orig_dir, input_dir))
                target_path = os.path.join(target_dir, orig_base + ".mp4")
            os.makedirs(target_dir, exist_ok=True)
            try:
                make_video(input_paths, target_path, fps=fps)
            except Exception as e:
                print(f"Error: {e} in {root}: Ignoring this folder")

def convert_to_gif_recursive(input_dir, output_dir, fps=10):
    # Recursive search for 'mp4 files' and convert them to gif in output_dir, maintaining the folder structure
    for root, dirs, files in tqdm(os.walk(input_dir)):
        files = [file for file in files if file.endswith('.mp4')]
        if len(files) >= 1:
            files = natsorted(files)
            for file in files:
                input_path = os.path.join(root, file)
                orig_base = os.path.splitext(file)[0]
                target_dir = os.path.join(output_dir, os.path.relpath(root, input_dir))
                target_path = os.path.join(target_dir, orig_base + ".gif")
                os.makedirs(target_dir, exist_ok=True)
                try:
                    make_gif(input_path, target_path, fps=fps)
                except Exception as e:
                    print(f"Error: {e} in {root}: Ignoring this folder")


def concat_images(images, row_size, col_size):
    # Concatenate multiple images
    assert len(images) <= row_size * col_size, f"Too many images: {len(images)} > {row_size * col_size}"
    h, w = images[0].size
    img = Image.new('RGB', (w * col_size, h * row_size))
    for i, image in enumerate(images):
        img.paste(image, (i % col_size * w, i // col_size * h))
    return img

def attach_ext(path, ext):
    if not path.endswith(ext):
        path += ext
    return path

def save_tensor(input_tensor, output_path, is_grayscale=False, save_type="images", fps=10, row_size=None, col_size=None):
    dim = input_tensor.dim()
    shape = input_tensor.shape
    type_4d = ("images", "video", "gif", "cat_image")
    type_5d = ("cat_images", "cat_video")
    if save_type in type_4d:
        if is_grayscale:
            # Normalize tensor
            input_tensor = (input_tensor - input_tensor.min()) / (input_tensor.max() - input_tensor.min())
            images = torch_to_pil_batch(input_tensor, is_grayscale=True)
        else:
            images = torch_to_pil_batch(input_tensor, is_grayscale=False)
    elif save_type in type_5d:
        raise NotImplementedError(f"Invalid save_type: {save_type}")
    else:
        raise ValueError(f"Invalid save_type: {save_type}")

    if save_type == "images":
        for i, img in enumerate(images):
            if output_path.endswith(".png"):
                output_full_path = output_path[:-4] + f"_{i:03d}.png"
            else:
                output_full_path = os.path.join(output_path, f"{i:03d}.png")
            img.save(output_full_path)

    elif save_type == "video":
        output_path = attach_ext(output_path, ".mp4")  # attach extension if not exists
        save_video(images, output_path, fps=fps)

    elif save_type == "gif":
        output_path = attach_ext(output_path, ".gif")  # attach extension if not exists
        save_gif(images, output_path, fps=fps)

    elif save_type == "cat_image":
        num_imgs = len(images)
        if row_size is None and col_size is None:
            row_size = int(np.sqrt(num_imgs))
            col_size = int(np.ceil(num_imgs / row_size))
        if row_size is None:
            row_size = int(np.ceil(num_imgs / col_size))
        if col_size is None:
            col_size = int(np.ceil(num_imgs / row_size))
        if row_size * col_size < num_imgs:
            row_size = int(np.ceil(num_imgs / col_size))
        
        img = concat_images(images, row_size, col_size)
        output_path = attach_ext(output_path, ".png")  # attach extension if not exists
        img.save(output_path)
        
    elif save_type == "cat_images":
        raise NotImplementedError("Not implemented yet")
    elif save_type == "cat_video":
        raise NotImplementedError("Not implemented yet")
    else:
        raise ValueError(f"Invalid save_type: {save_type}")

def concat_videos(input_paths, output_path, fps=10, margin=0, fill_color=0):
    # Concatenate multiple videos
    readers = [imageio.get_reader(input_path) for input_path in input_paths]
    writer = imageio.get_writer(output_path, fps=fps)
    for frames in zip(*readers):
        if margin > 0: # add margin between videos (so no margin for the last video)
            frames = [np.pad(frame, ((0, 0), (0, margin), (0, 0)), mode='constant', constant_values=fill_color) for frame in frames[:-1]] + [frames[-1]]
        frame = np.concatenate(frames, axis=1)
        writer.append_data(frame)
    writer.close()

def concat_and_draw_text(
        input_paths,
        output_path,
        text,
        texts=None,
        font=cv2.FONT_HERSHEY_TRIPLEX,
        org1=(30, 30),
        color1=(0, 0, 0),
        thickness1=2,
        scale1=1,
        org2=(30, 30),
        color2=(0, 0, 0),
        thickness2=2,
        scale2=1,
    ):
    # Concatenate images and draw text
    imglist = [cv2.imread(input_path) for input_path in input_paths]
    if texts is not None:
        assert len(texts) == len(imglist)
        imglist_new = []
        for img, txt in zip(imglist, texts):
            org = (org2[0], img.shape[0] - org2[1])
            img = cv2.putText(img, txt, org, font, scale2, color2, thickness2, cv2.LINE_AA)
            imglist_new.append(img)
        imglist = imglist_new

    # Concatenate images
    img = np.concatenate(imglist, axis=1)

    (text_height, text_width), _ = cv2.getTextSize(text, font, scale1, thickness1)
    org = (org1[0], org1[1] + text_height)
    img = cv2.putText(img, text, org, font, scale1, color1, thickness1, cv2.LINE_AA)

    # Save image
    cv2.imwrite(output_path, img)

def label_video(input_path, output_path, label, font=cv2.FONT_HERSHEY_TRIPLEX, org=(30, 30), color=(0, 0, 0), thickness=2, scale=1):
    # Label a video
    reader = imageio.get_reader(input_path)
    writer = imageio.get_writer(output_path, fps=reader.get_meta_data()['fps'])
    for img in reader:
        (text_height, text_width), _ = cv2.getTextSize(label, font, scale, thickness)
        org = (org[0], img.shape[0] - org[1])
        img = cv2.putText(img, label, org, font, scale, color, thickness, cv2.LINE_AA)
        writer.append_data(img)
    writer.close()