## Automatic whole slide pathology image diagnosis framework via unit stochastic selection and attention fusion
- Author: Chen et al.
- Journal: Neurocomputing
- Year: 2022
- Link: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8786216/pdf/nihms-1667698.pdf


### Abstract
- Automatic pathology slide diagnosis is still a challenging task for researchers because of the high-resolution, significant morphological variation, and ambiguity between malignant and benign regions in whole slide images (WSIs). 
- In this study, we introduce a general framework to automatically diagnose different types of WSIs via unit stochastic selection and attention fusion. 
  - We first train a unit-level convolutional neural network (CNN) to perform two tasks: constructing feature extractors for the units and for estimating a unit’s non-benign probability. 
  - Then we use our novel stochastic selection algorithm to choose a small subset of units that are most likely to be non benign, referred to as the Units Of Interest (UOI), as determined by the CNN. 
- Next, we use the attention mechanism to fuse the representations of the UOI to form a fixed-length descriptor for the WSI’s diagnosis. 


### Introduction
- Because of the high-resolution characterization of the WSI, currently, it is almost impossible to process the WSI directly.
The standard procedure to analyze WSI includes the following steps: 
  - 1) splitting a WSI into multiple units. 
  - 2) performing unit level representation learning. 
  - 3) fusion of the representations of units to form a fixed-length WSI descriptor. 
  - 4) WSI diagnosis based upon the WSI descriptor

- In this study, we propose to select a subset of the suspicious units in a stochastic manner for the WSI feature fusion to avoid the feature attenuation issue. 
- We take advantage of the fine-tuned unit classification model, which can estimate units’ probabilities belonging to different categories. 
- The unit selection is mainly based on each unit’s non-benign probability, namely the slide’s probability of
not being benign. 
- Those units with high non-benign probabilities are more likely to be chosen, and we denote these selected units as Units of Interest (UOI).

- The main contributions in this paper are:
1. We propose a general framework for the WSI analysis. This framework can be applied to both histological and
cytological applications.
2. We introduce a novel unit stochastic selection algorithm for the WSI model training, aiming to focus on suspicious units in the slide and improve the robustness of the WSI model.
3. We adopt the attention model to capture the weights of the selected Units Of Interest (UOI) to form more discriminating WSI representation.
4. Extensive experiments on three different types of pathology slides demonstrate the generality and effectiveness of the proposed framework via various fusion methods

<img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/9924de0d-94ac-43bf-87ac-ab361b9b2398'>


### Dataset
- We successively collect two batches of thyroid slides. 
  - The first collected slides contain 114 benign, 50 uncertain, and 181 malignant slides. 
  - The second collected slides include 83 benign, 7 uncertain, and 165 malignant slides.

### Method
#### Processing Units Cropping
- The size of both histological and cytological slides images are is extremely large. 
- Typically each WSI could have a full spatial resolution larger than 50, 000 × 50, 000 pixels at ×40 magnification. 
- Currently, it is not feasible to directly take the gigapixel WSI as a whole and feed it into the deep neural network. 
  - The widely used manner is to split the WSI into multiple small units, individually process each unit, and then fuse units’ outcome for the final diagnosis. 
- After locating the tissue regions, we split the whole WSI with a fixed length of stride in both vertical and horizontal directions.
- We propose an approximation method to crop the processing unit in the cytopathology slide by detecting the center of each cell and setting a fixed size to crop all the cells instead of detecting cell boundaries. 
  - To conduct the detection of cell centers, we adapt the segmentation network U-Net and train the network by providing the mask of cell centers instead of the binary mask of cell regions

#### Unit Feature Learning with CNN
- For the thyroid tissue slides, we make the annotation in a relatively coarse manner by drawing broad contours to cover regions of the same category. 
- Then we crop the units from these drawn contour surrounded regions and give the units the same label as the contour’s.
- Instead of training the CNN classifier from scratch, we train the WSI unit-level feature extractor based on the million-scale ImageNet pre-trained model, which can provide better parameter initialization and meanwhile speed the training process.

#### UOI Selection
- From a normal-sized WSI, we can obtain more than one thousand processing units from a normal WSI. 
  - However, even for a malignant WSI, there exist large regions being benign. 
  - When fusing units’ features for a WSI, the features from benign units would weaken the overall WSI representation discrimination capability, especially when the benign region taking accounting the majority area in of the WSI. 
  - With the fine-tuned CNN model, we can estimate the probability of units belonging to different categories. 
    - For all pathology diagnoses, including binary and multi-classification, we can calculate the non-benign probability of 
 each unit by summing its probabilities belonging to all the non benign categories. 
- We propose to select a subset of UOI from all the cropped units of the WSI based on the unit’s non-benign probability.


#### Attention-based WSI Representation Fusion
- Instead of equally treating all selected UOI in the fusion process, we adopt the attention mechanism to learn different weights for the UOI, with the assumption that the unit with a higher weight tends to be more informative for the WSI diagnosis.
- k selected units used for the WSI representation, $f_1, ..., f_k$ represent features of these units, where $f)k \in \mathbb{R}_d$, and the number of diagnosis categories is c.
  - We set $W \in \mathbb{R}^{c \times 1}$ and $V \in \mathbb{R}^{d \times c}$ as the attention model's parameters.
  - The weight:
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/505dd60c-ef69-4ad5-ae18-eefe048fdef0' width=50%>
  - The WSI's representation can be calculated as: $f_{wsi} = \sum _{i=1}  ^k w_i f_i$.


### Experiment
#### Unit feature extraction
- We choose two most widely used neural networks, VGG16bn and ResNet50, to train unit-based classification model
for feature extraction. 
- We change the final fully-connected layer based on the number of diagnosis categories in the pathology application and fine-tune the ImageNet pre-trained model.

#### UOI selection
- Currently, we set the parameters in the UOI selection mainly based on the average number of units in the WSIs of a specific application.


#### WSI unit fusion
- We compare the average pooling fusion, which equally treats all selected UOI and average them to obtain the WSI descriptor, with the self-attention fusion in the experiments. 
- In addition, we compare with the concatenation of pooling features with attention features, termed as “concat” fusion manner.
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/19c7c530-c5cc-4094-9b50-9668a6e6952f'>

    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/d3f6afee-9c43-4d2e-bb2f-dd88413722da'>

