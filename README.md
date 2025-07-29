# Generative Adversarial Network

## Main Sections
- [Brief Introduction](#brief-introduction)
- [GAN Introduction](#what-is-gan)
- [My Implementation](#my-implementation)
- [Model Architecture](#model-architecture)
- [Results and Insights](#results)
- [Conclusion](#conclusion)

## Brief Introduction

- This is my mini project to learn about GAN.

## What is GAN?
- It is a type of ***neural network*** that is used to ***generate*** content (images, music, speech ...).

### Motivation
- An adversarial game between two models.
- A **generative model** (G) generates content, then an adversary: a discriminative model (D) tries to guess if the content (whether a real sample or a generated sample) is from a model or the data.
- G gets better by fooling D, and D gets better by catching fakes.

### Adversarial Nets
- Both mentioned models are ***multilayer perceptrons***.
- The training process is pretty straightforward, a minimax algorithm:
	- D: maximize the probability of assigning the correct label to both the training samples from the data and the generated samples from G.
	- In contrast to D, we train G to minimize the likelihood of its content getting "caught".
 - We now have the following equation:
```math
\min _G \max _D V(D, G)=\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}(\boldsymbol{x})}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]
```
where:

- $`V(D, G)`$ is the **value function** of the adversarial game.  
- $`\mathbb{E}_{\boldsymbol{x}\sim p_{\text{data}}(\boldsymbol{x})}[\;\cdot\;]`$ denotes the **expectation** over real data samples $x$ drawn from the true data distribution $`p_{\text{data}}(x)`$.  
- $`\mathbb{E}_{\boldsymbol{z}\sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\;\cdot\;]`$ denotes the **expectation** over noise vectors $z$ drawn from a simple prior distribution $`p_z(z)`$ (e.g.\ Gaussian or uniform).  
- $`D(\boldsymbol{x})\in(0,1)`$ is the **Discriminator’s** estimated probability that sample $`x`$ is real.  
- $`G(\boldsymbol{z})`$ is the **Generator’s** output (a “fake” sample) when fed noise $`z`$.  
- $`\log D(\boldsymbol{x})`$ encourages $`D`$ to assign high probability to real data.  
- $`\log\bigl(1 - D(G(\boldsymbol{z}))\bigr)`$ encourages $`D`$ to assign low probability to generated (fake) data, and by minimizing this term $`G`$ learns to **fool** $`D`$. 

## My Implementation

- In this repository, I will try to explore as many GAN architectures as possible.
- Currently existing architectures:
    - [DCGAN](dcgan.md)
    - [CycleGAN](cyclegan.md)