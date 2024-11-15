


import os
import sys
import torch

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from utils import show_anns
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator



if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")


np.random.seed(seed = 2024)


# load model
sam2_checkpoint = "path/to/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "path/to/configs/sam2.1/sam2.1_hiera_l.yaml"
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)


# load image
image_path = 'path/to/image'
image = Image.open(image_path)
image = np.array(image)


# generate mask
masks = mask_generator.generate(image)

save_path = 'path/to/save'

plt.figure(figsize=(20, 20))
plt.imshow(image)
ax, img = show_anns(masks)
ax.imshow(img)
plt.axis('off')
plt.savefig(os.path.join(save_path, 'sam2_mask.png'))
plt.show() 



# 마스크 크기와 맞는 바운딩박스 생성
bboxes = []
for mask in masks:
    bbox = mask['bbox']
    # bbox = [x, y, w, h]
    bboxes.append(bbox)

bboxes = np.array(bboxes)
bbox_areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
bbox_areas = bbox_areas.astype(int)



#  이미지 상 바운딩 박스 그리기 
plt.figure(figsize=(20, 20))
plt.imshow(image)
for bbox in bboxes:
    x, y, w, h = bbox
    plt.plot([x, x+w, x+w, x, x], [y, y, y+h, y+h, y], color='r', linewidth=2)
plt.axis('off')
plt.savefig(os.path.join(save_path, 'sam2_bbox.png'))
plt.show()



# 각각의 바운딩박스 저장, file name :  {image_name} _ 01, 02, 03, ... 순서로 저장
for i, bbox in enumerate(bboxes):
    x, y, w, h = bbox
    cropped = image[y:y+h, x:x+w]
    cropped = Image.fromarray(cropped)
    cropped.save(os.path.join(save_path, f'{i+1}.png'))


