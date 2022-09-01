## An image is worth 16x16 words: Transformers for image recognition at scale
- Authors : Dosovitskiy, Alexey, et al.
- Journal : arXiv preprint
- Year : 2020
- Link : https://arxiv.org/pdf/2010.11929.pdf

### Abstract
- In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. 
- We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. 
- When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks, Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.

### Introduction
- Self-attention-based architectures, in particular Transformers, have become the model of choice in NLP.
- In computer vision, convolutional architectures remain dominant. Inspired by NLP successes, multiple works try combining
CNN-like architectures with self-attention, some replacing the convolutions entirely.
-  The latter models have not yet been scaled effectively on modern hardware accelerators due to the use of specialized attention patterns. ➔ classic ResNet-like architectures are still SOTA.
- Applying a standard Transformer directly to images, with the fewest possible modifications. 
➔ We split an image into patches and provide the sequence of linear embeddings of these patches as an input to a Transformer.
- When trained on mid-sized datasets such as ImageNet without strong regularization, these models yield modest accuracies of a few percentage points below ResNets of comparable size.
- Transformers lack some of the inductive biases inherent to CNNs, such as translation equivariance and locality, and therefore do not generalize well when trained on insufficient amounts of data.
- We find that large scale training trumps inductive bias. `ViT` attains excellent results when pre-trained at sufficient scale and transferred to tasks with fewer datapoints.

### Method
- We follow the original Transformer (Vaswani et al., 2017) as closely as possible.

#### `Vision transformer (ViT)`
  <img src="https://user-images.githubusercontent.com/57218700/185126901-b3a1c7b6-5517-4fcb-ae39-5e5a67d5f3d6.png" width=80%>
  <img src="https://user-images.githubusercontent.com/57218700/185127516-8bb6d77b-6d92-4283-82ab-0aa6aa14b976.png" width=80%>

- The standard Transformer receives as input a 1D sequence of token embeddings. To handle 2D images, we reshape the image x ∈ R<sup>H×W×C</sup> into a sequence of flattened 2D patches x<sub>p</sub> ∈ R<sup>N×(P<sup>2</sup>·C)</sup>.
- (H, W) is the resolution of the original image, C is the number of channels, (P, P) is the resolution of each image patch, and N = HW/P<sup>2</sup> is the resulting number of patches.
- We prepend a learnable embedding to the sequence of embedded patches, whose state at the output of the Transformer encoder serves as the image representation y.
- We use standard learnable 1D position embeddings
  ➔ The Transformer encoder consists of alternating layers of multiheaded selfattention and MLP blocks. Layernorm (LN) is applied before every block, and residual connections after every block.
  
- **Inductive bias**    
  ➔ Vision Transformer has much less image-specific inductive bias than CNNs. only MLP layers are local and translationally equivariant, while the self-attention layers are global. The two-dimensional neighborhood structure is used very sparingly.
  
- **Hybrid architecture**    
➔ As an alternative to raw image patches, the input sequence can be formed from feature maps of a CNN.

- **Fine tuning and higher resolution**    
➔ We pre-train ViT on large datasets, and fine-tune to (smaller) downstream tasks.  It is often beneficial to fine-tune at higher
resolution than pre-training. When feeding images of higher resolution, we keep the patch size the same, which results in a larger effective sequence length.    
➔ The pre-trained position embeddings may no longer be meaningful. We therefore perform 2D interpolation of the pre-trained position embeddings.

### Experiments
- ViT performs very favourably, attaining state of the art on most recognition benchmarks at a lower pre-training cost. Lastly, we perform a small experiment using self-supervision.
- **Datasets** : ImageNet with 1.3M, ImageNet-21k with 14M and JFT with 303M
- **Model Variants** : ResNet and Hybrids (feed the intermediate feature maps into ViT with patch size of one pixel)
- Comparison with SOTA
  <img src="https://user-images.githubusercontent.com/57218700/187826354-47039dba-ea05-4ef7-9431-67eb45ee866f.png" width=80%>

### Conclusion
- Unlike prior works using self-attention in computer vision, we do not introduce image-specific inductive biases into
the architecture apart from the initial patch extraction step.
- Strategy works surprisingly well when coupled with pre-training on large datasets.
- Relatively cheap to pre-train.