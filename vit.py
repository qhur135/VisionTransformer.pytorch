import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat # einops 차원 변환
from einops.layers.torch import Rearrange,Reduce
from torchsummary import summary

img = Image.open('./cat.jpg') # test image
fig = plt.figure()
plt.imshow(img)

# resize to imagenet size
transform = Compose([Resize((224,224)), ToTensor()]) # compose - 여러 단계 변환하는 경우, 여러 단계 묶을 수 있음
x = transform(img) # img (224,224)로 사이즈 바꾸고, 텐서로 바꿈
x = x.unsqueeze(0) # 첫번째 차원에 1인 차원 추가 add batch dim # shape:[1,3,224,224]

# image flatten - multiple patches
patch_size = 16 # 16 pixels
patches = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)

# PatchEmbedding
class PatchEmbedding(nn.Module): # 상속
    # 이미지 입력받아 패치 사이즈로 나누고, 1차원 벡터로 projection. 그 결과에 cls token, positional encoding 추가
    # 변수와 함수 정의, 초기화
    def __init__(self, in_channels: int=3, patch_size: int=16, emb_size: int=768, img_size: int=224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size), # performance gain
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size)) # 랜덤
        self.positions = nn.Parameter(torch.randn((img_size//patch_size)**2+1, emb_size))
    # init에서 정의된 것들을 사용
    def forward(self, x:Tensor) -> Tensor:
        b, _, _, _ = x.shape # b : 1
        x = self.projection(x) # 1차원 벡터로 projection
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b) # token b times copy - 확장
        x = torch.cat([cls_tokens, x], dim=1) # cls token 맨앞에 붙이기
        x += self.positions # add position embedding
        return x

print(PatchEmbedding()(x).shape)


