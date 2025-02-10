## Interactive thyroid whole slide image diagnostic system using deep representation
- Author: Chen et al.
- Journal: Computer methods and programs in biomedicine
- Year: 2020
- Link: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7492444/pdf/nihms-1607574.pdf


### Abstract
- We develop an interactive whole slide diagnostic system for thyroid frozen sections based on the suspicious regions preselected by pathologists.
- We propose to generate feature representations for the suspicious regions via extracting and fusing patch features using deep neural networks. We then evaluate region classification and retrieval on four classifiers and three supervised hashing methods based on the feature representations.
- We evaluate the proposed system on 345 thyroid frozen sections and achieve 96.1% cross-validated classification accuracy, and retrieval mean average precision (MAP) of 0.972.



### Introduction
- The dimension of a WSI larger than 50,000 by 50,000 pixels is quite common. When directly handling the WSI, even the loading and pre-processing takes more than 10 secs, and current existing diagnosis systems can hardly finish a diagnosis within one minute. 
- Additionally, WSIs are created through a series of procedures, thereby potentially generating low-quality WSIs. 
  - Tissues in WSI can be bent, wrinkled; dust may also contaminate the slides. Other factors, such as blurring and severe color variation, may also affect the AI diagnosis.

- Based on the practical diagnosis experience from expert pathologists, the suspicious regions in most thyroid WSI are not hard to identify even under the low-power field in microscopy, and the number of required suspicious regions for careful examination is usually less than five.

- We propose an interactive thyroid WSI diagnostic system to support the diagnosis of pathologists. In the whole slide image viewer, the pathologists first inspect and draw a suspicious region, which can be either rectangle or contour. 
- Based on pre-selected suspicious regions, we construct deep representations based on the CNN feature extractor. 
  - With the ROI representation, the system provides classification and retrieved similar regions to pathologists for references.

### Dataset
- In total, we collect and annotate 345 frozen section thyroid slides, including 114 benign, 50 uncertain, and 181 malignant samples.

### Method
#### Slides collection and annotation
- Thyroid frozen sections are categorized as benign, uncertain, or malignant, based on the requirements for developing
surgery plans.
- Pathologists annotate not only malignant regions, but also annotate benign and uncertain regions for the evaluation of ROI-based classification and retrieval.

    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/c2417be2-4789-4e6d-9cbb-8b3c3c10bc57' width=50%>

#### Patch-based feature extraction
- As the large size of the thyroid WSI, the annotated ROI might be larger than 10,000 × 10,000 pixels.
  - These factors can affect classification performance if taking ROI as a whole. Following the widely used analysis manner of the WSI, we adopt the patch-based scheme to generate a representation for the ROI, such that a patch classifier needs to be
firstly trained for patch feature extraction.
- **Patch Cropping**
  - After pathologists’ annotation on WSIs, we have annotated ROIs with category labels
  - we randomly crop patches from ROI based on its minimum enclosing rectangle and set the label of the cropped patch to be the same as the ROI. 
  - We crop the patch with a size of 256×256 loading from level 2 of WSI, corresponding to a patch image of 1024×1024 in level 0 of WSI.
- **Patch Classifier**
  - A large number of patches of these three different categories are produced after the patch cropping from ROIs. 
  - Because of CNN’s state-of-the-art performance in image recognition, we employ the CNN model for the patch classification.


#### ROI representation generation
- The CNN model combines representation learning with a fully connected layer to perform the classification task. 
  - With the labels as supervision, parameters in the CNN are tuned through multiple epoch training. We take the output of the penultimate layer, namely the layer before the fully connected layer, for thyroid patch representation.
  - With the trained patch feature extraction model, the ROI representation generation can be divided into three steps: 
    - 1) splitting ROI bounding rectangle into multiple patches; 
    - 2) extracting features for all valid patches. 
    - 3) fusing patches’ features to generate a fixed-length descriptor for the ROI. 
  - In the patch sampling process, we take a self-overlapping patch cropping manner.
  - We only keep those patches with at least 75% pixels inside the ROI.
- After putting the patches from ROIs into the CNN model, we obtain multiple patch feature vectors. 
  - As the size of ROIs is different, the number of feature vectors differs among ROIs. Here in this study, as the ROI is preselected, thus the appearance inside ROI is roughly consistent. 
  - There will not have a significant difference among patches from the same ROI.
- Therefore we propose all patches to have the same weight and use power mean with exponent p of all patch features to generate the representation for an ROI, which is formulated as:
  - $R_{rep}^p$: ROI's representation, n: the number of valid patches in a ROI, k: the number of elements in $f_i$.
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/55d7e60a-5d02-4792-ae92-765f34f40b76' width=30%>


#### **ROI Classification**
- With the patch fused deep representation for the ROI, we explore and compare four widely used classifiers for ROI diagnosis, which include decision tree (DT), support vector machine (SVM), random forest (RF), and multilayer perceptron (MLP). 

#### **ROI Retrieval**
- Based on the fixed-length representation and label for each ROI, we propose to employ supervised hashing methods to retrieve ROIs which have the same labels and close content with the query ROI.
- In supervised hashing with kernels (KSH), the ROI representation is firstly mapped to high dimensional space with kernel functions and then projected into binary codes. 
  - The pairwise labels between different ROI representations are adopted to improve the discrimination of the binary codes.
  - Supervised discrete hashing (SDH) formulates hashing objective function by introducing an auxiliary variable and thus can be effectively solved by a regularization algorithm. 
    - Column sampling based discrete supervised hashing (COSDISH) directly learns the discrete hashing code from semantic label information. Specifically, it iteratively samples several columns from the semantic similarity matrix and decomposes the
hashing code into two parts to be alternately optimized discretely.


