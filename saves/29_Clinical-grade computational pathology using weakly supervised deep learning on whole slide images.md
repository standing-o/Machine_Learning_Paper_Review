## Clinical-grade computational pathology using weakly supervised deep learning on whole slide images
- Author: Campanella et al.
- Journal: Nature medicine
- Year: 2019
- Link: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7418463/

### Abstract
- We evaluated this framework at scale on a dataset of 44,732 whole slide images from 15,187 patients without any form of data curation. 
- Tests on prostate cancer, basal cell carcinoma and breast cancer metastases to axillary lymph nodes resulted in areas under the curve above 0.98 for all cancer types. Its clinical application would allow pathologists to exclude 65–75% of slides while retaining 100% sensitivity. 

### Introduction
- One of the main contributions of this work is the scale at which we learn classification models. We collected three datasets in the field of computational pathology: (1) a prostate core biopsy dataset consisting of 24,859 slides; (2) a skin dataset of 9,962 slides; and (3) a breast metastasis to lymph nodes dataset of 9,894 slides. 
- The slides collected for each tissue type represent the equivalent of at least 1 year of clinical cases and are thus representative of slides generated in a true pathology laboratory, including common artifacts, such as air bubbles, microtomy knife slicing irregularities, fixation problems, cautery, folds and cracks, as well as digitization artifacts, such as striping and
blurred regions. 
- To fully leverage the scale of our datasets, it is unfeasible to rely on supervised learning, which requires manual annotations. 
  - Instead, we propose to use the slide-level diagnosis, which is readily available from anatomic pathology laboratory information systems (LISs) or electronic health records, to train a classification model in a weakly supervised manner.
  - Crucially, diagnostic data retrieved from pathology reports are easily scalable, as opposed to expert annotation for supervised learning, which is time prohibitive at scale. To be more specific, the slide-level diagnosis casts a weak label on all tiles within a particular WSI.
  - In addition, we know that if the slide is negative, all of its tiles must also be negative and not contain tumor. In contrast, if the slide is positive, it must be true that at least one of all of the possible tiles contains tumor. This formalization of the WSI classification problem is an example of the general standard multiple instance assumption.
- Current methods for weakly supervised WSI classification rely on deep learning models trained under variants of the MIL assumption. Typically, a two-step approach is used, where first a classifier is trained with MIL at the tile level and then the predicted scores for each tile within a WSI are aggregated, usually by combining (pooling) their results with **various strategies, or by learning a fusion model**. 
- Inspired by these works, we developed a novel framework that leverages MIL to train deep neural networks, resulting in a semantically rich tile-level feature representation. 
  - These representations are then used in a recurrent neural network (RNN) to integrate the information across the whole slide and report the final classification result.


### Dataset
- We collected three large datasets of hematoxylin and eosin-stained digital slides for the following tasks: (1) prostatic carcinoma classification; (2) BCC classification; and (3) the detection of breast cancer metastasis in axillary lymph nodes. 
- Each dataset was randomly divided at the patient level in training (70%), validation (15%) and test (15%) sets. 

### Method
#### MIL-based slide diagnosis.
- Classification of a whole digital slide (for example, WSI) based on a tile-level classifier can be formalized under the classic MIL approach when only the slide-level class is known and the classes of each tile in the slide are unknown. 
- Each slide $s_i$ from our slide pool S can be considered a bag consisting of a multitude of instances (we used tiles of size 224 × 224 pixels). 
  - For positive bags, there must exist at least one instance that is classified as positive by some classifier. 
  - For negative bags, instead, all instances must be classified as negative. 
  - Given a bag, all instances are exhaustively classified and ranked according to their probability of being positive. 
  - If the bag is positive, the top-ranked instance should have a probability of being positive that approaches 1; 
  - if it is negative, its probability of being positive should approach 0. 
- Solving the MIL task induces the learning of a tile-level representation that can linearly separate the discriminative tiles in positive slides from all other tiles.
- This representation will be used as input to an RNN. The complete pipeline for the MIL classification algorithm comprises the following steps: (1) tiling of each slide in the dataset (for each epoch, which consists of an entire pass through the training data); (2) a complete inference pass through all of the data; (3) intraslide ranking of instances; and 4) model learning based on the top-ranked instance for each slide.

##### Slide Tiling
- The instances were generated by tiling each slide on a grid. Otsu’s method is used to threshold the slide thumbnail image to efficiently discard all background tiles, thus drastically reducing the amount of computation per slide. 
  - Tiling can be performed at different magnification levels and with various levels of overlap between adjacent tiles. We investigated three magnification levels (5×, 10x and 20×). 

#### Model Training
- This is the most stringent version of MIL, but we can relax the standard MIL assumption by introducing hyper-parameter K and assume that at least K tiles exist in positive slides that are discriminative. 
  - For K = 1, the highest ranking tile in bag $B_{s_i}$ is then $b_{i,k}$. The output of the network $y_i = f_θ (b_i, k)$ can then be compared to $y_i$ , the target of slide $s_i$, through the cross entropy loss.
- We used mini-batches of size 512 for AlexNet, 256 for ResNets and 128 for VGGs and DenseNet201. All models were initialized with ImageNet pretrained weights. 

#### Naive multiscale aggregation
- Given models $f_{20×}, f_{10x}$, and $f_{5x}$ trained at 20×, 10× and 5× magnifications, a multiscale ensemble can be created by pooling the predictions of each model with an operator. We used average and max-pooling to obtain naive multiscale models.

#### Random forest-based slide integration.
- Given a model f trained at a particular resolution, and a WSI, we can obtain a heat map of tumor probability over the slide. We can then extract several features from the heat map to train a slide aggregation model. For example, Hou et al. used the count of tiles in each class to train a logistic regression model. 
- Here, we extend that approach by adding several global and local features, and train a random forest to emit a slide diagnosis. The features extracted are: (1) total count of tiles with probability ≥0.5; 
  - (2–11) tenbin histogram of tile probability; 
  - (12–30) count of connected components for a probability threshold of 0.1 of size in the ranges 1–10, 11–15, 16–20, 21–25, 26–30, 31–40, 41–50, 51–60, 61–70 and >70, respectively; 
  - (31–40) ten-bin local histogram with a window of size 3 × 3 aggregated by max-pooling ......

#### RNN-based slide integration.
- Model f mapping a tile to class probability consists of two parts: a feature extractor $f_F$ that transforms the pixel space to representation space, and a linear classifier $f_C$ that projects the representation variables into the class probabilities. 
- The output of $f_F$ for the ResNet34 architecture is a 512-dimensional vector representation. 
- Given a slide and model f, we can obtain a list of the S most interesting tiles within the slide in terms of positive class probability.
- With S = 1, the model does not recur and the RNN should learn the $f_C$ classifier. This approach can be easily extended to integrate information at multiple scales.
  - We obtain the S most interesting tiles from a slide by averaging the prediction of the three models on tiles extracted at the same center pixel but at different magnifications
- In all of the experiments, we used 128 dimensional vectors for the state representation of the recurrent unit, ten recurrent steps (S = 10), and weighted the positive class to give more importance to the sensitivity of the model. All RNN models were trained with cross-entropy loss and SGD with a batch size of 256.

#### Visualization of feature space.
- For each dataset, we sampled 100 tiles from each test slide, in addition to its top-ranked tile. Given the trained 20× models, we extracted for each of the sampled tiles the final feature embedding before the classification layer. We used t-distributed stochastic neighbor embedding (t-SNE) for dimensionality reduction to two dimensions.
-  t-SNE visualization of the representation space for the BCC and axillary lymph node models.

<img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/ecb46aae-0e83-4925-9fdb-226aab68cbf0'>


### Results
#### Test performance of Resnet34 models trained with MIL for each tissue type
- We trained `ResNet34` models to classify tiles using MIL. At test time, a slide is predicted positive if at least one tile is predicted positive within that particular slide. This slide-level aggregation derives directly from the standard multiple instance assumption and is generally referred to as max-pooling. 
- Simple ensemble models were generated by max-pooling the response across the different magnifications. 
- We note that these naive multi-scale models outperformed the single-scale models for the prostate dataset in terms of accuracy and area under the curve (AUC), but not for the other datasets. 
- Models trained at 20× achieved AUCs of 0.986, 0.986 and 0.965 on the test sets of the prostate, BCC and axillary lymph node datasets, respectively, highlighting the efficacy of the proposed method in discerning tumor regions from benign regions in a wide variety of tissue types.

#### Dataset size dependence of classification accuracy
- We conducted experiments to determine whether the dataset was large enough to saturate the error rate on the validation set. Although the number of slides needed to achieve satisfactory results may vary by tissue type, we observed that, in general, at least 10,000 slides are necessary for good performance.

#### Model introspection by visualization of the feature space in two dimensions.
- To gain insight into the model’s representation of histopathology images, we visualized the learned feature space in two dimensions so that tiles that have similar features according to the model are shown close to each other. 
- The prostate model shows a large region of different stroma tiles at the center of the plot, extending towards the top right corner. 
  - The top left corner is where benign-looking glands are represented. The bottom portion contains background and edge tiles. The discriminative tiles with high tumor probability are clustered in two regions at the bottom and left of the plot. A closer look reveals the presence of malignant glands. Interestingly, a subset of the top-ranked tiles with a tumor probability close to 0.5, indicating uncertainty, are tiles that contain glands suspicious of being malignant.

<img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/079e77a7-4789-4425-bafd-c371f1a2461f'>


#### Comparison of different slide aggregation approaches
- The max-pooling operation that leads to the slide prediction under the MIL assumption is not robust. 
- A single spurious misclassification can change the slide prediction, possibly resulting in a large number of false positives. 
  - One way to mitigate this type of mistake is to learn a slide aggregation model on top of the MIL classification results. For example, Hou et al. learned a logistic regression based on the number of tiles per class as predicted by an ensemble of tile classifiers. 
  - Similarly, Wang et al. extracted geometrical features from the tumor probability heat map generated by a tile-level classifier and trained a random forest model, winning the CAMELYON16 challenge. Following this, we trained a random forest model on manually engineered features extracted from the heat map generated by our MIL-based tile classifier. 
  - The previous aggregation methods do not take advantage of the information contained in the feature representation learned during training. Given a vector representation of tiles, even if singularly they were not classified as positive by the tile classifier, taken together they could be suspicious enough to trigger a positive response by a representation-based slide-level
classifier. 
  - Based on these ideas, we introduce an RNN-based model that can integrate information at the representation level to emit a final slide classification. Interestingly, information can also be integrated across the various magnifications to produce a multi-scale classification. At 20×, the MIL-RNN models resulted in 0.991, 0.989 and 0.965 AUCs for the prostate, BCC and breast metastases datasets, respectively. 
  - For the prostate experiment, the MIL-RNN method was statistically significantly better than max-pooling aggregation. 

