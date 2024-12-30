from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from model import Generator  

class ImagePatcher:
    def __init__(self, patch_width=256, patch_height=256):
        self.patch_width = patch_width
        self.patch_height = patch_height

    def create_patches(self, image):
        patches = []
        img_width, img_height = image.size

        for y in range(0, img_height, self.patch_height):
            for x in range(0, img_width, self.patch_width):
                patch = image.crop((x, y, x + self.patch_width, y + self.patch_height))
                patches.append(patch)

        return patches


class ImageRestitcher:
    def __init__(self, patch_width=256, patch_height=256):
        self.patch_width = patch_width
        self.patch_height = patch_height

    def restitch_image(self, patches, original_size):
        img_width, img_height = original_size
        full_image = Image.new('RGB', (img_width, img_height))

        idx = 0
        for y in range(0, img_height, self.patch_height):
            for x in range(0, img_width, self.patch_width):
                full_image.paste(patches[idx], (x, y))
                idx += 1

        return full_image


class ImageProcessor:
    def __init__(self, generator_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        
        # Initializing the generator model
        self.generator = Generator()  
        
        # Loading the state dictionary
        try:
            state_dict = torch.load(generator_path, map_location=self.device)
            self.generator.load_state_dict(state_dict)  
            self.generator = self.generator.to(self.device)
            self.generator.eval()  
        except Exception as e:
            raise RuntimeError(f"Failed to load the generator model: {str(e)}")
        
        self.patcher = ImagePatcher()
        self.restitcher = ImageRestitcher()

    def preprocess_patch(self, patch):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) 
        ])
        return transform(patch).unsqueeze(0).to(self.device)

    def postprocess_patch(self, tensor):
        tensor = tensor.squeeze(0).permute(1, 2, 0).detach().cpu()
        return ((tensor + 1) / 2.0 * 255).clamp(0, 255).numpy().astype(np.uint8)

    def colorize_image(self, image):
        patches = self.patcher.create_patches(image)

        colorized_patches = []
        for patch in patches:
            patch_tensor = self.preprocess_patch(patch)
            with torch.no_grad():
                colorized_tensor = self.generator(patch_tensor)
            colorized_patch = self.postprocess_patch(colorized_tensor)
            colorized_patches.append(Image.fromarray(colorized_patch))

        return self.restitcher.restitch_image(colorized_patches, image.size)
