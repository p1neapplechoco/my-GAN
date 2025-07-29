# CycleGAN
- **CycleGAN** is a type of **generative adversarial network** (GAN) designed for **unpaired image‑to‑image translation**. Unlike a DCGAN, which generates images from random noise using stacked convolutional layers, a CycleGAN learns to transform images from one domain into another— for example, converting photos of horses into zebras and vice versa—without requiring paired training examples.

## The Task
- We will implement unpaired image‑to‑image translation for two domains:
  1. **Artistic style transfer**: Convert photographs into the style of Picasso’s paintings—and back again.
  2. **Time‑of‑day conversion**: Transform daytime scenes into nighttime atmospheres—and vice versa.
- This setup lets us train two cycle‑consistent generators—one for each direction in each domain—without requiring paired examples.

## Why CycleGAN?
- **Unpaired, bidirectional translation**: CycleGAN uses two generators—one to map images from domain A to B and another to map from B back to A—so it can learn both directions without any paired examples.
- **Cycle‑consistency constraint**: A cycle‑consistency loss forces each original image to be recoverable after a round‑trip translation (A→B→A or B→A→B), ensuring that content and structure are preserved.
- **Dual discriminators**: Two discriminators (one per domain) guide the generators to produce outputs indistinguishable from real images in each style.
$\rightarrow$ Together, these components form a “cycle” of translations that both captures stylistic characteristics and guarantees reversibility—hence the name CycleGAN.

## Framework
- We’ll use [PyTorch](https://docs.pytorch.org/docs/stable/index.html) for model definition and training.

## Dataset
### Picasso
- Have not yet found a suitable dataset.

### Time-of-day conversion
- We will be using the [Unpaired Day and Night cityview images](https://www.kaggle.com/datasets/heonh0/daynight-cityview) from Kaggle.

## Model Architecture
**Generator**: 
```python
class Generator(nn.Module):
    def __init__(self, n_blocks=6):
        super().__init__()
        
        # Initial convolution + 2 downsampling blocks
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        ]
        
        # Residual blocks
        for _ in range(n_blocks):
            layers.append(ResidualBlock(256))
        
        # 2 upsampling blocks + final convolution
        layers += [
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh(),
        ]
        
        self.model = nn.Sequential(*layers)

```

consists of `n_blocks` of ResidualBlock:
```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)
```
- Input: a 3-channel RGB image.
- Output: a 3-channel RGB image whose style is shifted.
  
**Discriminator**
```python
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4
            nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output layer
            nn.Conv2d(512, 1, 4, stride=1, padding=1, bias=False)
            # (No sigmoid: we’ll use a least‑squares or hinge loss.)
        )

```
## Results

### Picasso 


### Day Night Shifter


## Conclusion