## A visual-language foundation model for computational pathology
- Author: Lu, Ming Y et al.
- Journal: Nature Medicine
- Year: 2024
- Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC11384335/pdf/nihms-2015817.pdf


### Abstract
- `CONCH` stands for CONtrastive learning from Captions for Histopathology.
  - It is a visual-language foundation model aimed at enhancing digital pathology through the integration of images and text.
- Developed with over 1.17 million image-caption pairs derived from various histopathology images and biomedical text.
- Utilizes task-agnostic pretraining, which means it can be applied to a variety of downstream tasks without needing extensive retraining.
- Evaluated on 14 diverse benchmarks in histopathology.
- Achieves state-of-the-art performance in various tasks including:
  - Image classification, Segmentation, Captioning, Retrieval (text-to-image and image-to-text)
- The use of a single pretrained model allows for zero-shot classification, meaning it can classify images without task-specific training data.
- Demonstrates robustness in applying to tasks like cancer subtyping, tissue classification, and more.
- Can facilitate machine learning workflows with little to no supervised fine-tuning required.
- **Methodology**
  - Combines an image encoder, a text encoder, and a multimodal fusion decoder.
  - Trained using methods that align image and text representations and also aimed at caption prediction, enhancing its understanding of the data.

### Dataset
- All internal data (WSIs, pathology reports, EMRs) was de-identified before analysis.
- No direct patient involvement or recruitment; informed consent was waived for archival pathology slides.
- Created a specific dataset called **PMC-Path**, combined with EDU dataset to create a full pretraining dataset of 1,786,362 image–caption pairs.
  - Filtered out nonhuman and special staining data, resulting in 457,372 valuable human histopathology pairs.

### Visual-language Pre-training
- **Objective** |  Combine image and text representations for better understanding and classification of histopathology data.
- **Method** | Utilize an equal-weighted combination of image–text contrastive loss and captioning loss inspired by the CoCa model.
- **Components of the Model**
  - **Image Encoder** $(f·; \theta)$
    - **Backbone** | Vision Transformer (ViT) architecture with: 12 transformer layers, 12 attention heads, 768 embedding dimensions, 3,072 hidden dimensions and 16x16 tokens.
    - Transforms raw RGB pixel values into semantically rich feature maps.
    - Attention Poolers: fcontrast; θcontrast: Computes a global image token using 1 query to summarize overall image representation.
fcaption; θcaption: Generates 256 image tokens using 256 queries for local details (needed for caption generation).
  - Text Encoder $(g · ; ϕ)$
    - Processes text data.
    - Also consists of 12 transformer layers and embedding and hidden dimensions same as image encoder.
  - **Multimodal Text Decoder** $(h · ; ψ)$
    - Generates text based on visual and textual inputs.
- Attention Mechanism
  - Backbone (ViT): Generates a semantic representation of images.
  - Attentional Pooler Modules:
    - First Pooler ($f_{contrast} · ; θ_{contrast}$) | Computes a global representation of the image using 1 query.
    - Second Pooler ($f_{caption} · ; θ_{caption}$) | Produces 256 image tokens to capture fine-grained details.
- **Training Mechanics**
  - **Mini-Batch** | Composed of $M$ image-caption pairs $(x_i, w_i)$.
  - **Objective Function**
$$
    \mathcal{L} = - \frac{1}{2M} \sum_{i=1}^{M} \log \frac{\exp(\tau u_i^T v_i)}{\sum_{j=1}^{M} \exp(\tau u_i^T v_j)} - \frac{1}{2M} \sum_{j=1}^{M} \log \frac{\exp(\tau v_j^T u_i)}{\sum_{i=1}^{M} \exp(\tau v_j^T u_i)} - \frac{1}{M} \sum_{i=1}^{M} \sum_{t=1}^{T+1} \log p(w_{i,t} | w_{i,0:t-1}, x_i; \theta, \phi, \psi)
$$
- $u_i$ | Output from the image encoder for the \(i^{th}\) image.
- $v_i$ | Output from the text encoder for the \(i^{th}\) caption.
- $T$ | The number of tokens in the caption.
- $\tau$ | A scaling factor that adjusts the similarity score.
- The first two terms maximize the similarity between paired images and captions while minimizing the similarity with negative pairs. 
- The last term maximizes the likelihood of correct captioning given the image.

### Pretraining unimodal encoders
- Prior research indicated that self-supervised pretraining of unimodal modules using unpaired data before joint visual-language pretraining can enhance downstream zero-shot transfer performance.
- **Image Encoder Pretraining**
  - The image encoder was pretrained using "iBOT," a leading self-supervised pretraining method for unlabeled image data.
  - An internal dataset included 16 million 256x256 image tiles extracted from the tissue regions across 21,442 Whole Slide Images (WSIs) from 350 cancer subtypes (OncoTree classification)
- Language Model Pretraining
  - A diverse corpus for pretraining consisted of: 
    - Pathology educational texts.
    - Final diagnosis sections from over 550,000 surgical pathology reports.
    - Over 400,000 histopathology-relevant PubMed abstracts.
- Objective Function
  - The loss function $\mathcal{L}_{clm}$ used during training can be expressed as:
  $$
  \mathcal{L}_{clm}(\xi) = - \sum_{t=1}^{T+1} \log p(w_t | w_{0:t-1}; \xi)
  $$
  - Here, $w_t$ is the token being predicted at position $t$.
  - $w_{0:t-1}$ are the previously generated tokens.
  - $xi$ represents the parameters of the autoregressive model.

### Zero-shot transfer on ROIs and tiles
- Zero-shot transfer
  - A method used to classify images without needing any labeled examples for those classes. In this process, pre-defined text prompts are used to identify image classes.
- Text Prompts
  - Each class is linked to a descriptive text prompt. For instance, the class “adenocarcinoma” could be represented as “this is adenocarcinoma.” This helps in embedding the class descriptions into a vector space.
- Embedding Calculation
  - Each class prompt results in an l2-normalized embedding $vj$, which quantifies the prompt in a multi-dimensional space using a text encoder trained on a specific dataset.
  - The embeddings are used as linear classifier weights.
- Image Processing
  - Each image is also converted into an l2-normalized embedding $ui$.
  - The similarity between an image and each class embedding is computed using cosine similarity:
  $$
    y_i = \arg\max_j (u_i^T v_j)
  $$
  - $y_i$ is the predicted class for image $i$.
  - $u_i^T$ is the transpose of the image embedding vector, and $v_j$ is the class embedding vector.
- Evaluation Metrics
  - Balanced accuracy | This averages the accuracy across classes, ensuring each class contributes equally regardless of its size.
F1 Score: A measure of a test's accuracy considering both precision and recall.
  - Quadratic Cohen’s κ | Used for evaluating classification accuracy, especially in cases like prostate Gleason grading, where misclassifications between adjacent classes carry less penalty.
- Cross-modal Retrieval
  - The same zero-shot computation can be modified for image-to-text and text-to-image retrieval applications by measuring how images and text relate in the latent space, evaluating using Recall at specific "K" values (how many of the top retrieved results are correct).


### Results
#### Zero-shot classification of diverse tissues and diseases
- The study presents a model called CONtrastive learning from Captions for Histopathology (`CONCH`) that leverages contrastively aligned visual-language pretraining.
- **Zero-Shot Learning**
  - The model can perform tasks without additional labeled training data, meaning it can classify images based on previously unseen categories directly.
  - This is contrary to the traditional method that requires a new model for each classification task.
- **Performance**
  - Even if zero-shot classification isn't always perfect for clinical use, `CONCH` showed surprising effectiveness in certain tasks, providing a baseline performance in situations where labeled data is scarce.
- **Process**
  - Classes are represented by predetermined text prompts (e.g., "invasive lobular carcinoma").
  - Classification occurs by matching the image with the most similar text prompt in the model's representation space.
Multiple phrasings for the same diagnosis are utilized to enhance accuracy.

- **Datasets and Tasks**
  - The model was evaluated on multiple tasks including:
    - Slide-level tasks: e.g., breast cancer and lung cancer classifications.
    - ROI-level tasks: e.g., colorectal cancer tissue classification.
  - The model achieved high accuracy (e.g., 90.7% for non-small-cell lung cancer subtyping).

- **Evaluation Metrics**
  - The main metric used was balanced accuracy for unbalanced datasets.
  - For subjective tasks, Cohen's κ scores were employed to evaluate model agreement with human pathologists.

- **Comparison**
  - `CONCH` outperformed other state-of-the-art models (like PLIP and BiomedCLIP) in slide-level and ROI-level tasks.
  - Example accuracies:
    - 90.7% for NSCLC subtyping.
    - 91.3% for BRCA subtyping, compared to lower scores from other models.

- **Human Interpretability**
  - The model supports generating heatmaps that show the relationship between image tiles and text prompts, aiding in understanding the model's decisions.

#### Few-shot classification with task-specific supervised learning
- **Purpose**
  - The research examines how a specific model, `CONCH` can improve classification tasks in histopathology by using fewer labeled training examples compared to previous models.
- **Zero-shot Capability**
  - This model can recognize diverse tissues and diseases without needing extensive training data. 
  - It can be applied to various tasks without collecting and annotating new data for training each time.
- **Few-shot Learning**
  - While zero-shot is efficient, some tasks may still benefit from a few labeled examples to enhance performance. 
  - The goal is to find how few labels might be needed for optimal results.
- **Efficient Training**
  - The researchers tested different amounts of training labels per class (nc), evaluating performance on various tasks while minimizing the number of labels needed for effective classification.
- **Experimental Results**
  - In slide-level tasks, `CONCH` achieved significant accuracy improvements over traditional models like ResNet50, particularly in subtyping tasks such as BRCA and NSCLC.
  - In ROI-level tasks, `CONCH` showed comparable performance to other state-of-the-art encoders like CTransPath while requiring fewer labels.
  - `CONCH`'s performance remained competitive even in zero-shot tasks against supervised learning methods for specific classifications.

- **Label Efficiency**
  - The research indicates that using `CONCH`, fewer training labels are needed to achieve high accuracy compared to models like PLIP and BiomedCLIP, demonstrating `CONCH`'s effective label efficiency.
  - Future Implications
    -  The ability of `CONCH` to perform competently in zero-shot settings encourages its use as a strong baseline when developing supervised learning models, especially for tasks with limited labeled data.

#### Zero-shot segmentation
- WSIs can be extremely large (gigapixels) and contain varied tissue types, cell shapes, and structures. This diversity can complicate the task of identifying specific areas within these images.
- This involves identifying distinct regions in a WSI based on the characteristics of interest, which can help reduce the number of image tiles required for further analysis.
- Labeling data at the sub-slide level is costly and time-consuming. Thus, a model that can segment slides without labeled examples (in a zero-shot manner) is highly beneficial.
- **Method Used**
  - The model segments a WSI by dividing it into smaller tiles.
  - Each tile is classified using zero-shot classification techniques.
  - The predicted label for each tile is assigned to all pixels in that tile.
  - To avoid abrupt changes at the edges of neighboring tiles, a 75% overlap between tiles is maintained. Prediction scores in overlapping regions are averaged to create a smoother overall segmentation map.

- **Model Evaluation**
  - The model was assessed using datasets SICAP (for prostate tumor vs. normal tissue) and DigestPath (for malignant vs. benign tissues in colorectal cancer specimens).
  - Performance metrics included: Dice Score, Precision, Recall

-  In the SICAP dataset, the model (CONCH) achieved: Average Dice(0.601), Average Recall(0.751), Average Precision(0.672)
- In the DigestPath dataset, it achieved: Average Dice(0.615), Average Recall(0.709), Average Precision(0.663)
- `CONCH` performed better than other models like PLIP and BiomedCLIP in both datasets, demonstrating its effectiveness even in a zero-shot context.

### Discussion
- Traditional computational pathology tools mainly focus on image data and structured patient information, often neglecting valuable textual descriptions that can help in understanding diverse pathology cases.
- Recent studies have tried using image and caption data from various sources, but their performance in real-world applications (like zero-shot classification) remains unsatisfactory, especially in tasks involving rare diseases.
- The study introduces a large histopathology image-text dataset (1.17 million examples) that leads to the creation of a high-performing visual-language foundation model. This model can perform various clinical tasks such as:
  - Classification of diseases, Retrieval of relevant cases, Segmentation of tissues

- **Zero-shot Recognition**
  - The developed model demonstrates strong zero-shot recognition capabilities, suggesting it can alleviate the need for extensive annotated training data. 
  - In fact, it surpasses traditional supervised learning performance in certain scenarios, particularly in few-shot learning environments.

- The improved capabilities for image-to-text and text-to-image retrieval can assist pathology trainees, physicians, and researchers in finding relevant examples effectively.
- Although promising, the study notes that pretrained models struggle with complex classification scenarios and tasks involving a multitude of classes, particularly with rare diseases.
- The research discusses additional experiments on data filtering, pretraining algorithms, and the potential for integrating visual-language models with traditional supervised learning approaches, especially for well-studied diseases.

- **Limitations**
  - The scale of data used for pretraining is much smaller compared to other large-scale models, suggesting room for improvement.
  - Challenges remain with managing data overlap between training and testing datasets and ensuring robustness across varying imaging conditions and techniques.