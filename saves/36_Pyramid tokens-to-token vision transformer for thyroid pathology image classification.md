## Pyramid tokens-to-token vision transformer for thyroid pathology image classification
- Author: Yin et al.
- Journal: IEEE
- Year: 2022
- Link: [https://ieeexplore.ieee.org...](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9784139&casa_token=S_alHpg-F4wAAAAA:9OeV9J7lsngP7_GwnrvQYQnDeeLVUySi4evysp34vrKBJwHeN7kxZ5VgD0aou_87FNLT-fZK&tag=1)


### Abstract
- The first difficulty is that although the method based on the CNN framework achieves a high accuracy rate, it increases the model parameters and computational complexity. 
  - The second difficulty is balancing the relationship between accuracy and model computation. 
  - It makes the model maintain and improve the classification accuracy as much as possible based on the lightweight.
-  We propose a new lightweight architecture called Pyramid `Tokens-to-Token VIsion Transformer (PyT2T-ViT)` with multiple
instance learning based on Vision Transformer. 
- We introduce the feature extractor of the model with Token-to-Token ViT (T2TViT) to reduce the model parameters.
  -  The performance of the model is improved by combining the image pyramid of multiple receptive fields so that it can take into account the local and global features of the cell structure at a single scale.


### Introduction
- To address the challenge of model parameters, we choose a lightweight `Tokens-to-Token Vision Transformer (T2T-ViT)` instead of CNN.
- The image blocks obtained after WSI preprocessing are used as the input of `T2T-ViT`, and the images are converted from 3D data to serialized data, thereby establishing a global relationship. 
- To address the challenge of model performance, we combine image pyramids with `T2T-ViT` to build a `Pyramid
Tokens-to-Token Vision Transformer (PyT2T-ViT)` for thyroid cancer diagnosis.


### Dataset
- Our thyroid dataset is composed of 560 clinical cases.  
- The cancer genome atlas (TCGA), but partial slides with an image size of approximately 2048×1536 pixels.


### Methodology
- In this section, we propose a new lightweight classification framework that combines multiple receptive fields to predict
bag-level relationships. 
- This model is divided into pyramidlevel tokens to token module (PyT2T module) and transformer encoder backbone.

    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/d8f7882b-ec24-4213-bd23-c3df9d4086a5'>


#### `PyT2T` Module
- **`T2T` Module**
  - The `PyT2T` module we proposed contains three layers, each level of the pyramid is a separate ViT module, and bags are used as the input of the hierarchical structure. 
  - In `T2T-ViT`, multiple iterations of tokes are performed to avoid redundancy, and each iteration includes Soft Spilt (SS) and Re-structurization
  - Soft Spilled aims to prevent information errors in the iterative process and use a specific receptive field size to overlap and cut patches. 
  - The output image size after re-structurization can be obtained by the calculation formula of convolution.
  - Although it is similar to the convolution operation, the multiplication and addition calculation is not performed.
  - Here MSA represents a self-attention layer with alternating multiple heads, and MLP represents a multilayer perceptron
with layer normalization.

- Image Pyramid
  - After many iterations, the original image $I_0$ becomes a reorganized image In, and it passes through Soft Spilt again to get a certain layer of fixed-length tokens in `PyT2T`.
  - The model connects the receptive fields of a specific size used by Soft Spilt in each layer. 
  - We compare the layer-by-layer tokenization process to a pyramid
  - The higher the level, the smaller the perception field and the more the model can focus on the overall characteristics.
  - Finally, add the fixed-length tokens $P_i$ in each layer of the pyramid model, and add them together to be the output $P
^{all}$ of the `PyT2T` module: $P^{all} = \sum_{i=1}  ^n  P_i$.
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/1bb2537f-7e8a-4a2b-9c7d-38dd710405dc' width=50%>


#### Transformer encoder backbone
-  According to the advantages and characteristics of convolutional neural networks, we build a deep-narrow architecture based on baglevel. 
  - Narrow refers to increasing the number of layers to enhance feature richness, and deep refers to reducing the
channel dimension and the feature redundancy. 
  - The size of this model is smaller than traditional CNN, and the accuracy rate has been significantly improved.
- We also use a class token to connect to it and add Sinusoidal Position Embedding (PE) to maintain the position information of patch embeddings.

   <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/0955ec8b-1910-458d-a1f5-5aac34dc6dfa' width=50%>

   <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/67c628b5-7e79-46ea-a436-5fe65f40d297' width=50%>