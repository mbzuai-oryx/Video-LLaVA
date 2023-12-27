import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image
from ram.models import ram


class TaggingModule(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        import gc
        self.device = device
        image_size = 384
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # load RAM Model
        self.ram = ram(
            # pretrained='checkpoints/ram_swin_large_14m.pth',
            # pretrained='/share/users/shehan/VideoGrounding/checkpoints/ram_swin_large_14m.pth',
            pretrained = "grounding_evaluation/weights/ram_swin_large_14m.pth",
            image_size=image_size,
            vit='swin_l'
        ).eval().to(device)
        print('==> Tagging Module Loaded.')
        gc.collect()

    @torch.no_grad()
    def run_on_pil_image(self, original_image):
        print('==> Tagging...')
        img = self.transform(original_image).unsqueeze(0).to(self.device)
        tags, tags_chinese = self.ram.generate_tag(img)
        print('==> Tagging results: {}'.format(tags[0]))
        return [tag for tag in tags[0].split(' | ')]
    
    @torch.no_grad()
    def run_on_video(self, video_pil_list):
        
        tags_in_video = []
        
        for frame_img in video_pil_list:
            frame_tensor = self.transform(frame_img).to(self.device)
            tags, _ = self.ram.generate_tag(frame_tensor.unsqueeze(0), threshold=0.95)
            tags_in_video.append(tags[0].split(' | '))

        
        return tags_in_video

from collections import defaultdict
string_counts = defaultdict(int)

def get_unique_tags(tags_in_split):
    # Iterate through the list of lists and count occurrences
    for sublist in tags_in_split:
        for string in sublist:
            string_counts[string] += 1

    # Convert the defaultdict to a regular dictionary
    string_counts_dict = dict(string_counts)

    # Print the resulting dictionary
    # for key, value in string_counts_dict.items():
    #     print(f'{key}: {value}')
    
    sorted_keys = sorted(string_counts_dict, key=lambda k: string_counts_dict[k], reverse=True)
    print(sorted_keys)
    
    return sorted_keys
