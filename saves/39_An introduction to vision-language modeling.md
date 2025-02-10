# An introduction to vision-language modeling
- Author: Bordes et al.
- Journal: arXiv preprint
- Year: 2024
- Link: https://arxiv.org/pdf/2405.17247

## Abstract
- `Vision-Language Models (VLMs)` extend Large Language Models (LLMs) to incorporate visual information alongside textual data.
- **Applications**
  - `VLMs` can serve as visual assistants to help navigate environments.
  - They can generate images based on high-level text descriptions.
- **Challenges**
  - There are significant challenges in improving the reliability of `VLMs`.
  - Language works in discrete terms, while vision operates in a complex, higher-dimensional space making it difficult to translate concepts directly between the two.


## Introduction
- Recent advancements in language modeling, such as Llama and ChatGPT, have made these models highly capable in various tasks.
- Originally focused on text, these models are now capable of processing visual inputs, leading to new applications.
- Despite progress, challenges persist in linking language to vision, such as:
  - Difficulty understanding spatial relationships.
  - Issues with counting and additional data annotation needs.
  - Lack of comprehension for attributes and ordering.
  - Problems with prompt engineering and hallucination in outputs.



## The families of `VLMs`
- `VLMs (Vision-Language Models)` aim to connect computer vision and natural language processing by utilizing deep learning techniques, particularly transformers.

- **Training Paradigms**
  - 1. **Contrastive Training** uses pairs of examples (positive and negative). The model learns to create similar representations for positive pairs and different representations for negative pairs.
  - 2. **Masking** involves reconstructing masked portions of data.
  - 3. **Generative** `VLMs` can produce images or text (captions) themselves. They are usually more expensive to train due to the complex nature of their functionality.


<img src='https://github.com/user-attachments/assets/8d1fdb91-3055-499a-b001-845db335a528'>
<img src='https://github.com/user-attachments/assets/cf52c37a-31e6-4974-838c-4ed0909a09ee'>


- **Pretrained Backbones**
  - Often employs existing large language models (such as Llama) to create a connection between image encoders and text encoders.
  - This method is typically less resource-intensive than training models from scratch.

- **Early Works**
  - Bidirectional Encoder Representations from Transformers (**BERT**) (2019)
  - **visual-BERT, ViL-BERT**: Integrates visual data with text for better comprehension.
  - These models are trained using two main tasks:
    - **Masked modeling task** involves predicting missing parts of the input, helping the model learn what information might be absent.
    - **Sentence-Image prediction task** predicts whether a textual caption describes the content of an image, helping it develop a connection between language and visuals.


### 1. Contrastive-based `VLMs`
#### **Energy-Based Models (EBMs)**
- EBMs train a model $E_\theta$ that assigns low energy to observed data and high energy to unseen data.
- The goal is to differentiate between real data (which should have low energy) and noise or unobserved data (which should have higher energy).
- The **energy function** is defined as $E_\theta (x)$, where $x$ is the input data and $\theta$ are the parameters of the model.
- **Boltzmann Distribution**
  - The probability density function (the probability of the input $x$ under the model) is given by:
    $p_\theta (x) = \frac{e ^{- E_\theta (x)}}{Z_\theta}$
, where $E_\theta (x)$ is the energy of input $x$ and $Z_\theta = \sum_x e^{-E_\theta (x)}$ is the normalization factor ensuring $p_\theta (x)$ sums to 1 over all $x$.

#### **Maximum Likelihood Objective**
- The training objective is to minimize the discrepancy between model predictions and real data:
$arg \min_\theta E_{x \sim P_D} (x) [- \log p_\theta (x)]$
- **Gradient of the gradient**
  - The gradient of the objective is computed as follows:
   $\frac{\partial E_{x \sim P_D}(x)[-\log p_\theta(x)]}{\partial \theta} = E_{x^+ \sim P_D}(x) \frac{\partial E_\theta(x^+)}{\partial \theta} - E_{x^- \sim P_\theta}(x) \frac{\partial E_\theta(x^-)}{\partial \theta}$,
  - where
    - $x^{+} \sim P_D (x)$ = samples from the real data distribution.
    - $x^{-} \sim P_D (x)$ = samples from the model's distribution.
    - The first term adjusts the model to fit the real data while the second term helps in differentiating from negative samples.

#### **Noise Contrastive Estimation (NCE)**
- Relies on sampling from a noise distribution to approximate the model distribution:
- NCE defines a binary classification problem:
  - Predict 1 for real data C=1 and 0 for noise C=0. 
  - Loss function for NCE:
  $L_{NCE}(\theta) := - \sum_{i} \log P(C_i = 1 | x_i; \theta) - \sum_{j} \log P(C_j = 0 | x_j; \theta)$,
  - where $x_i$ = samples from real data distribution.
  - $x_j \sim p_n (x)$ = samples from noise distribution (often drawn from some random noise process).


#### Contrastive Language–Image Pre-training (CLIP)
- To learn a shared representation of images and their corresponding text (captions).
- Training Method:
  - Uses InfoNCE loss as a contrastive learning mechanism.
  - Positive Examples: Pairs of an image and its correct caption.
  - Negative Examples: The same image with all other captions from different images in the mini-batch.
- Shared Representation Space:
  - CLIP maps images and captions into a similar vector space, allowing them to be processed together.
- Training Dataset:
  - Initially trained on 400 million caption-image pairs sourced from the internet.
- Performance:
  - CLIP shows excellent zero-shot classification capabilities. This means it can classify images into categories it hasn't explicitly been trained on.
  - A ResNet-101 CLIP matched the performance of a supervised ResNet model, achieving 76.2% zero-shot classification accuracy.
- Variants:
  - SigLIP uses a different loss function (NCE loss with binary cross-entropy).
    - Performs better in zero-shot scenarios with smaller batch sizes compared to CLIP.
  - Latent Language Image Pretraining (Llip) focuses on accommodating diverse captioning for images.
    - Incorporates a cross-attention module to better connect image encodings to varied captions, improving performance in classification and retrieval tasks.

### 2. Masking
- **Masking** is a technique used to prevent certain data points from influencing the output in models.
- Relation to Denoising Autoencoder
  - Similar to denoising autoencoders, masking involves predicting missing parts in data that has a spatial structure.
- It is connected to image inpainting strategies where portions of an image are reconstructed.

#### Masked Language Modeling (MLM)
- Introduced by BERT (2019), a model that predicts missing words in a sentence using a masked approach, which is effective for transformer networks.

#### Masked Image Modeling (MIM)
- Examples include MAE (2022) and I-JEPA (2023), which apply the masking strategy to image representation learning.

#### `VLMs` with Masking Objectives:
- FLAVA (2022)
  - Leverages masking techniques to learn representations from both text and images through a structured training approach.
  - Includes separate encoders for images and text, which utilize masking during training, allowing for multi-layered fusion of data.
  - Achieves state-of-the-art results across multiple sensor modalities.
- MaskVLM (2023)
  - Focuses on reducing dependencies on pre-trained models by applying masking directly in the pixel and text token spaces, facilitating information flow between modalities.

#### Information Theoretic Perspective on `VLM`
- Discusses how `VLMs` can efficiently encode information by solving a rate-distortion problem, which involves maximizing the relevance of learned representations while minimizing wasted information.
- Introduces a mathematical formulation to model this relationship:
 $\text{arg min}_{p(z|x)} I(f(X); Z) + \beta \cdot H(X|Z)$,
  - where
    - $I(f(X); Z)$: Mutual information measuring the relevance between the input data $f(X)$ and the representation $Z$.
    - $\beta$: A trade-off parameter that determines the influence of the second part.
    - $H(X|Z)$: Conditional entropy representing the uncertainty of data X given the learned representation Z.
- A further related equation that bounds the objective is
$L = - \sum_{x \in D} E_{p(f)} p(Z|f(x)) [\log q(z) + \beta \cdot \log q(x|z)]$,
  - where
    - $q(z)$: Represents the distribution of the learned representation.
    - $D$: A dataset used for generating the representations.
- This equation emphasizes balancing between obtaining meaningful representations and retaining pertinent details from the original input.


### 3. Generative-based `VLMs`
- Unlike previous models that primarily work with latent representations (i.e., abstract features), generative models directly generate text and/or images.

#### CoCa
- **CoCa** learns a complete text encoder and decoder for tasks like image captioning.
- Loss Functions: Uses a new generative loss alongside contrastive loss to enable new multimodal understanding tasks without needing additional modules.
- Pretraining: Utilizes datasets like ALIGN (1.8 billion images with alt-text) and JFT-3B (29,500+ classes treated as alt-text).

#### CM3Leon
- A foundational model for text-to-image and image-to-text generation.
- **Tokenization**: Uses special tokens for modality shifts, enabling interleaved processing of text and images.
- **Training Process**
  - **Stage 1 (Retrieval-Augmented Pretraining)**: Uses a CLIP-based encoder to augment the input sequence with relevant multimodal documents and performs training via next token prediction.
  - **Stage 2 (Supervised Fine-tuning)**: Involves multi-task instruction tuning allowing content generation and processing across modalities, improving performance on a range of tasks.


#### **Chameleon**
- Introduces mixed-modal foundation models for generating and reasoning with intertwined text and non-text content.
- **Architecture**: Unified architecture from the start, using a token-based representation for both modalities.
- **Early-Fusion Strategy**: Maps both text and image modalities in a shared representational space from the beginning, allowing robust generation and reasoning capacities while addressing optimization challenges through novel architectural and training techniques.

#### Using generative text-to-image models for downstream vision-language tasks
- Recent advancements in models like Stable Diffusion and Imagen allow these systems to create images conditioned on text prompts.
- Typically known for generating images, these models can also perform classification and caption prediction without needing to be retrained specifically for those tasks.
- These models estimate $p_\theta (x|c)$, which represents the likelihood of generating image $x$ given text prompt $c$.
- **Classification via Bayes' Theorem**
  - When given an image $x$ and a set of text classes $(c_i)^n _{i=1}$, the model can classify the image based on Bayes' theorem:
    $p_\theta(c_i | x) = \frac{p(c_i) p_\theta(x | c_i)}{\sum_{j} p(c_j) p_\theta(x | c_j)}$,
  - where
    - $p_\theta (c_i | x)$: Posterior probability of class $c_i$ given image $x$.
    - $p (c_i)$: Prior probability of class $c_i$
    - $p_\theta(x | c_i)$: Likelihood of image $x$ occurring given class $c_i$.
    - The denominator sums the likelihoods across all classes, normalizing the probabilities.
- Generative Classifiers: This approach, known as "analysis by synthesis", links back to foundational techniques like Naive Bayes and Linear Discriminant Analysis, which classified data based on generative models.
- **Tokenization for Autoregressive Models**
  - To apply autoregressive modeling techniques effectively to images, images must first be tokenized into discrete tokens $(t_1,t_2,...,t_K)$.
    $\log p_\theta(x | c_i) = \sum_{j=1}^{K} \log p_\theta(t_j | t_{<j}, c_i)$
- **Image Tokenizer**
  - A common approach is to use the Vector Quantized-Variational AutoEncoder (VQ-VAE), which combines an auto-encoder and a vector quantization layer to discretize images.
  - Improvements in modern tokenizers may include perceptual and adversarial losses to capture finer details.
- **Diffusion Models**
  - These models estimate noise in images, making classification computationally expensive but effective in terms of performance.

- **Likelihood estimation with diffusion models**
  - Diffusion models are used for generating images but estimating their likelihood (density) is complex. Instead of directly estimating $p_\theta (x|c)$ (the probability of an image given a condition $c$), these models estimate noise $\epsilon$ in a noisy image $x_t$.
  - The classification techniques developed for these models focus on approximating a lower bound for the conditional image likelihood.
  - **Key equation**
      $\log p_\theta(x | c_i) \propto -E_{t,\epsilon} \left\|\epsilon - \epsilon_\theta(x_t, c_i)\right\|^2$,
    - where
      - $\log p_\theta (x|c_i)$: The log likelihood of the image $x$ given class $c_i$.
      - $E_{t, \epsilon}$:  Expected value over time $t$ and noise $\epsilon$, used to average the estimation errors.
      - $|| \cdot || ^2$: Represents the squared L2 norm, which measures the difference between the predicted noise $\epsilon_\theta$ and the actual noise $\epsilon$.
  - **Challenges**
    - Estimating the likelihood requires sampling multiple times to get a reliable Monte Carlo estimate, which can significantly increase computational costs. This challenge is exacerbated as the number of classes increases.
  - **Generative Classifiers**
    - Despite being computationally intensive, generative classifiers offer greater "effective robustness," making them perform better in out-of-distribution scenarios compared to discriminative models (like CLIP).
    - These classifiers have enhanced shape bias and better align with human judgment.
    - They can also be jointly adapted with discriminative models using only unlabeled test data, improving performance across various tasks.


### `VLMs` from Pretrained Backbones
- Training `Vision-Language Models (VLMs)` is expensive, needing extensive computational resources (hundreds to thousands of GPUs) and large datasets (hundreds of millions of images and text pairs).
- To reduce costs, researchers focus on leveraging existing large-language models (LLMs) and visual feature extractors instead of building `VLMs` from scratch. 
- By utilizing trained models, researchers aim to learn a mapping between text and image modalities. This allows LLMs to respond to visual questions with fewer computing resources.

#### Frozen
- Frozen is a pioneering model connecting vision encoders to LLMs through a lightweight mapping network that transforms visual features into text embeddings.
- **Architecture**
  - Vision Encoder: NF-ResNet-50 is trained from scratch.
  - Language Model: A pretrained transformer (7 billion parameters) is kept "frozen" to preserve its pre-learned features.
  - Training Objective: Uses a simple text generation goal based on the Conceptual Captions dataset.
- Capabilities
  - Demonstrates rapid task adaptation and efficient binding of visual and linguistic elements, marking a pivotal development toward multimodal LLMs.

#### The example of MiniGPT
- MiniGPT-4 accepts both text and image inputs, producing text outputs via a simple linear projection layer to align image and text representations.
- Trained on large datasets (5 million image-text pairs) using only four A100 GPUs in a short time (10 hours).
- Instruction-tuning phase requires 400 training steps with highly-curated data.

#### Other popular models using pretrained backbones
- **Qwen-VL and Qwen-VL-Chat** align visual representations with LLM input spaces, using transformer layers for compression.
- A vision-language model **BLIP-2** that processes images to generate text using a lightweight component (Q-Former) trained for mapping image and text embeddings, leveraging pretrained models to speed up training.

--------------------------------------
## A Guide to VLM Training
- **Importance of Scaling** | Recent research has highlighted that increasing the capability (compute and data scale) of deep neural networks can significantly boost their performance.
- **Success with CLIP** | For instance, the CLIP model was trained on 400 million images and required immense resources, using between 256 to 600 GPUs over several days or weeks.
- **Data Curation Pipeline** | New studies have introduced the idea that data quality can surpass quantity. Specifically, effective data curation can yield better results compared to merely increasing model size.
- **Training Model Insights**
  - **Data Quality** | The quality of the dataset is vital. Good models are built on diverse and balanced datasets, with careful pruning to eliminate duplicates and irrelevant data.
  - **Grounding Techniques** |  Ensuring that the VLM accurately associates text with visual context is crucial. This involves methods such as using bounding boxes and negative captions.
  - **Human Preference Alignment** | Models must also align their outputs with expectations from human users through specific training techniques.
  - **OCR Techniques** | Enhancements in Optical Character Recognition (OCR) capabilities are discussed, considering how VLMs are often utilized for reading and translating text.
- **Common Fine-tuning Methods** | Various standard practices are employed in the fine-tuning stage to enhance the performance of VLMs.

<img src='https://github.com/user-attachments/assets/92aa2b4b-25b6-4cc3-a0ed-6f4efed7e798'>


### Training Data
- DataComp is a framework developed to assess the quality of pretraining datasets for `Vision-Language Models (VLMs)` like CLIP, focusing on creating effective image-text datasets for downstream tasks.
  - It offers various datasets, from small (1.28 million pairs) to very large (12.8 billion pairs), to evaluate performance on 38 tasks.
- Data pruning is highlighted as essential for training efficient VLMs.

- **Methods of Data Pruning**
  - **Heuristic Methods**
    - Unimodal Filters: Remove texts of low complexity, non-English texts, and images based on quality metrics.
    - Multimodal Filters: Use image classifiers to filter out image-text pairs lacking alignment.
  - **Bootstrapping Methods**
    - Use pretrained VLMs to rank images and texts, discarding poorly aligned pairs.
    - **CLIP-Score**: Evaluates pairs based on cosine similarity of their embeddings in CLIP.
  - **Creating Diverse Datasets**
    - Sample from higher-quality datasets like ImageNet, aiming for balance in representation.
- **Diversity and Generalization**
  - A balanced dataset supports better generalization capabilities. Various sampling methods ensure coverage of diverse concepts.
- **Challenges**
  - While efforts aim for balance, perfect balance in datasets is impractical due to imbalances in web data distribution.
- **Zero-shot Performance**
  - The model's ability to perform tasks it hasn't been trained for heavily depends on the training data's variety related to those tasks.