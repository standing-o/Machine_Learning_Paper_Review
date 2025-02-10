## Pathologist-level interpretable whole-slide cancer diagnosis with deep learning
- Authors: Zhang et al.
- Journal: Nature Machine Intelligence
- Year: 2019
- Link: https://www.nature.com/articles/s42256-019-0052-1

### Abstract
- We present a novel pathology whole-slide diagnosis method, powered by artificial intelligence, to address the lack of
interpretable diagnosis. 
- Moreover, using 913 collected examples of whole-slide data representing patients with bladder cancer, we show that our method matches the performance of 17 pathologists in the diagnosis of urothelial carcinoma. 

### Introduction
- Our method diagnoses a slide via region-level tumour detection, pixel-level morphological analysis of nuclear anaplasia and architectural abnormalities, and establishes slide-level diagnosis. 
  - Each process is powered by neural networks; their cascading progressively encodes enormous pixels into meaningful and compact representations. Instead of only predicting diagnosis labels, our method includes interpretability mechanisms to decode learned representations into rich interpretable predictions, which are understandable to pathologists. 
- The description generation module is trained using tissue images and associated diagnostic reports provided by pathologists, while the visual attention module learns spontaneously only by observing the visual-semantic correspondences from image pixels to annotated description words of these images in data.
  - For example, the system builds direct correspondence between the text ‘severe crowding of nuclei’ and image regions that exhibit crowded nuclei. During inference, the system is capable of interpreting its observation explicitly through its text and visual outputs.

<img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/3a053189-3739-46e5-bfd9-7faff375ce75'>


### Dataset
- Te slides were H&E stained and scanned with a ×40 objective. Te data set contains 913 patient-exclusive slides for non-invasive low-grade papillary urothelial carcinoma (102) and non-invasive or invasive high-grade papillary urothelial carcinoma (811), which are the most common types of urothelial carcinoma in clinics.
- For each slide they were asked to annotate at most eight tumour regions, which they believed to contain diagnostically useful information, and eight non-tumour regions. To construct the II-Image data set, we randomly sampled a set of
images with 1,024×1,024 resolution around the annotated tumour and non-tumour regions. 
- Each tissue image had a tumour region binary mask and was assigned with a label equivalent to the diagnosis label of its source slide. 
  - The ‘tumour’ label was given if the image had been sampled from tumour regions. Conversely, the ‘non-tumour’ label was given if sampled from non-tumour regions. Note that this label assignment was ‘coarse’, because a tissue image from
a slide of a class could exhibit pathological features of a different class.
  - To construct the III-Report data set, we selected 221 non-invasive HG and LG papillary urothelial carcinoma slides from train and test sets in total, and sampled 4,253 1,024×1,024 images. Two experienced pathologists provided a paragraph of pathology report descriptions for each image and each full paragraph was ended with one of four
suspected conclusions: normal, LG papillary urothelial carcinoma, HG papillary urothelial carcinoma or insufficient information. 
- The IV-Diagnosis data set was collected with the assistance of our trained `s-net` and `d-net`.
  - Each slide is represented as a bag of ROI embedded features;

### Tumour Detection
- The `s-net` conducts tumour detection by classifying each pixel as tumour or non-tumour, represented as a probability value. The architecture of `s-net` resembles U-net. 
  - The II-Image data set is used to train s-net. Because the tumour region is partially annotated, unannotated region pixels have unknown classes. 
  - To bypass this problem, during network training we compute the losses of `s-net` only for annotated region pixels, while ignoring unannotated pixels. 
  - At the inference time, processing a whole slide is decomposed into two steps: (1) divide the slide into computationally memory-affordable tiles and (2) detect tumours within each slide tile using `s-net`. 
  - As a result, `s-net` generates a probability map as the tumour region detection. 
  - Based on the tumour detection result, the system conducts the following steps to automatically select a set of diagnostically useful tissue images (termed ROIs in the main text) around detected tumours from the whole slide, denoted as {$R_1, ..., R_D$}.
- These ROIs are the inputs of `d-net`. 
  - First, to decide the sampling location, the system computes the average pixel probability of each tile; this step estimates which tile contains more tumours so that more ROIs can be sampled from this tile. The average probabilities of all tiles are then normalized so that they all sum to one; 
  - The normalized value × D is the number of ROIs needed to sample from a tile. We set D=200 empirically. A pixel is treated as a tumour pixel if its probability of being tumour is greater than 0.5 while the others are not considered as candidate pixels. Finally, using probabilities of tumour pixels as weights, the system simply applies weighted sampling to select central candidate pixels and crop ROIs.


### Celluar-level ROI Characterization
-  The `d-net` plays the core role in the system in characterizing ROIs, generating an interpretable diagnosis and encoding observed visual information. It is a composite neural network that can combine multi-modal information. 
- Specifically, it includes an image model to represent visual knowledge by encoding image pixels into feature maps. It also includes a language model to generate diagnostic descriptions and network visual attention. 
- We utilize the Inception-v3 CNN26 with initial weights from a pretrained model on the ImageNet data set. 
- Input images are resized to 256×256 (from the original 1,024×1,024); so a slide is seen at the ×10 objective by the network). This down sampling rate does not cause a loss of critical information, as confirmed by pathologists. The image model produces feature maps $V = [\mathbf{v}_1, ..., \mathbf{v}_s]^T \in \mathbb{R}^{2048 \times 6 \times 6}$ with its last convolution layer, that is 2048 feature maps where each has 6 x 6 resolution.
- To effectively model such report structure priors, we designed a language model with two long short-term memory (LSTM) modules to model the concepts and then descriptions. This design is inspired by the design of hierarchical LSTM.
  - The language model is conditioned on these visual features to generate descriptions. LSTM is a computation unit that holds a hidden state vector $\mathbf{h}_t$ and a memory state vector $\mathbf{m}_t$ at each time step to integrate spatiotemporal information.


### IV-Diagnosis Dataset
- To conduct slide-level diagnosis, `a-net` aggregates encoded information from all ROIs {$R_1, ..., R_D$} in a slide and establishes slide diagnosis.
  - For each of the slide ROIs, there are two types of information extracted from `d-net`: encoded features $\mathbf{f}^R$ and raw class probability $\mathbf{p}^R$. We extract feature maps at several layers of Inception-v3 (namely mixed10, mixed9, mixed8 and mixed7). Convolutional feature maps are averaged along the spatial dimension to obtain feature vectors, which are then concatenated together as the encoded feature of an ROI, that is, $\mathbf{f}_R \in \mathbb{R}^{6144}$.
  - All D ROIs in a whole slide are organized as $R = \{ (\mathbf{f}^R _1, \mathbf{p}^R _1), ...,  (\mathbf{f}^R _D, \mathbf{p}^R _D)\}$.
- We follow this step to process slides and organize the IV-Diagnosis dataset {$R_i, l_{R_i}$} <sub>train</sub> and {$R_i, l_{R_i}$} <sub>val</sub> for the training of `a-net`.
- **Slide Diagnosis**
  - The `a-net` is implemented as a three-layer fully connected neural network. It takes integrated ROI feature encodings and predicts slide cancer labels. We propose a stochastic feature sampling mechanism to effectively augment training data through random feature combination so as to improve the model generalization. Algorithm 1 describes the training details of `a-net` using the IV-Diagnosis dataset. At the inference stage, `a-net` repeats this stochastic sampling-and-prediction process. Note that the ground-truth label $l_{R_i}$ to compute the sampling probability $\mathbf{p}_{R_j}$ [l<sub>R<sub>i</sub></sub>] is unknown; we simply alternate l of all classes (that is, HG and LG) over multiple sampling. 
  - Ten repeats are performed in total. The final diagnosis is the maximum class probability response of the accumulated probability of the 10-time predictions.
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/e3871c4b-ebee-4157-ac4b-984bd7fcf855' width=70%>


### Results
- Our method is designed to be applicable to a wide variety of cancer types. Here, we have validated it on a large dataset containing 913 haematoxylin and eosin (H&E) stained whole slides from patients with bladder cancer. 
- Grading these carcinomas as being of low grade (LG) or high grade (HG) is essential for specified therapies. 
- The data set was cleaned and manually annotated by pathologists using several carefully designed procedures using our
developed web-based annotation programs. 
- The data set was split into 620 whole slides for training, 193 slides for validation and 100 slides for testing.
- ROI characterization is performed by analysing the tumour appearance, cell morphological patterns and so on, and these features are interpreted using natural language descriptions. 
  - The system describes a certain number of observed cellular features together with feature-aware attention maps to indicate what the network sees when describing each of these features. A strong interpretation is given regarding the type of visual information observed by the network. 
  - An attention map contains real-valued per-pixel weights to decide which pixels are more important for a given feature observation. The attention maps are visualized in a binary manner that is alpha-blended with the input image. 
- A high positive rate setting and a high true negative rate setting of the network configuration are highlighted. 
- `s-net` achieves a 94% true positive (tumour) recall rate and simultaneously maintains a 95.3% negative (non-tumour) recall rate. 
- To demonstrate the superiority of the results, we also compared the results obtained using `d-net` to those obtained using a well-known image-to-text translation method as the baseline. 
  - Our method is elaborately designed to enhance the effective combination of modules for learning from multi-modal diagnosis and report data in network training and it outperforms the baseline on both metrics.
  - Furthermore, the inner working of `d-net` is a direct translation from image pixels to report words. Beyond the image-to-text generation shown already, a trained d-net also supports text-to-image retrieval. 
  - This capability provides a solution for doctors to query reference tissue images from databases by simply giving wanted feature descriptions. The text-to-image retrieval evaluation is also an exact measure of the translation quality of `d-net`, because it indicates a failure to retrieve a normal tissue image given a diseased tissue query description and vice versa. 
  - The `d-net` encodes the visual-semantic information of ROIs in a low-dimensional feature vector
- The goal of `a-net` is to aggregate such diagnostic information together in all slide ROIs and establish a final diagnosis. 
  - a-net is implemented as a three-layer fully connected neural network that receives slide ROI encodings and predicts the probability of cancer classes. 
  - We demonstrate the effectiveness of `a-net` by examining the behaviours of its hidden layers using the t-distributed Stochastic Neighbour Embedding (t-SNE) visualization algorithm.

- **Results for the whole-slide diagnosis**
<img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/578fcff7-ddbe-4bf3-a03b-1ff8279b5c76'>


### Conclusion
- The proposed method interprets predictions through natural language descriptions and visual attention, which
are understandable to pathologists when conducting a second review and visual inspection. 
- We believe that our method has strong generalizability for learning complex tissue structures and cell patterns in different cancer types. 
- In addition, in the context of precision medicine, we acknowledge the diagnostic value of contextual clinical information of patients beyond pixel knowledge or one type of stain in isolation. 


