## Classifying and segmenting microscopy images with deep multiple instance learning
- Author: Kraus et al.
- Journal: Bioinformatics
- Year: 2016
- Link: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4908336/pdf/btw252.pdf



### Abstract
- We introduce a new neural network architecture that uses MIL to simultaneously classify and segment microscopy images with populations of cells. 
- We base our approach on the similarity between the aggregation function used in MIL and pooling layers used in CNNs. 
- To facilitate aggregating across large numbers of instances in CNN feature maps we present the Noisy-AND pooling function, a new MIL operator that is robust to outliers. 
- Combining CNNs with MIL enables training CNNs using whole microscopy images with image level labels. 


### Introduction
- The network is designed to produce feature maps for every output category, as proposed for segmentation tasks
in Long et al. (2014). 
  - We pose cellular phenotype classification as a MIL problem in which each element in a class-specific feature map (approximately representing the area of a single cell in the input space) is considered an instance an entire class specific feature map (representing the area of the entire image) is considered a bag of instances annotated with the whole image label. 
  - We explore the performance of several global pooling operators on this problem and propose a new operator capable of learning the proportion of instances necessary to activate a label.
- We present a unified view of the classical MIL approaches as pooling layers in CNNs and compare their performances. 
  - To facilitate MIL aggregation in CNN feature maps we propose a novel MIL method, ‘NoisyAND’, that is robust to outliers and large numbers of instances. 
  - We show that our model is capable of learning a good classifier for full resolution microscopy images as well as individual cropped cell instances, even though it is only trained using whole image labels. 
  - We demonstrate that the model can localize regions with cells in the full resolution microscopy images and that the model predictions are based on activations from these regions.


### Methods
#### Convolutional MIL model for learning cellular patterns
- We propose a CNN capable of classifying microscopy images of arbitrary size that is trained with only global image level labels. The weakly supervised CNN is designed to output class-specific feature maps representing the probabilities of the classes for different locations in the input image. 
  - The CNN produces an image level classification over images of arbitrary size and varying number of cells through a MIL pooling layer. 
  - Individual cells can be classified by passing segmented cells through the trained CNN or by mapping the probabilities in class specific feature maps back to the input space.


#### Pooling layers as MIL
- In a CNN, each activation in the feature map is computed through the same set of filter weights convolved across the input image. The pooling layers then combine activations of feature maps in convolutional layers. 
- We formulate the MIL layer in CNNs as a global pooling layer over a class specific feature map for class i referred to as the bag $p_i$.
  - Without loss of generality assume that the i-th class specific convolutional layer in a CNN computes a mapping directly from input images to sets of binary instance predictions $I \rightarrow (p_{i1},...,p_{iN}$.
  - It first outputs the logit values $z_{ij}$ in the feature map corresponding to instance j in the bag i.
  - We define the feature level probability of an instance j belonging to class i as $p_{ij}$ where $p_{ij} = \sigma (z_{ij})$ and r is the sigmoid function.
- The image level class prediction is obtained by applying the global pooling function $g(\cdot)$ over all elements $p_{ij}$.
  - The global average pooling function $g (\cdot)$ maps the instance space probabilities to the bag space s.t. the bag level probability for class i is defined by $P_i = g(p_{i1}, p_{i2}, ...)$.
- While the MIL layer learns the relationship between instances of the same class, the co-occurrence statistics of instances from different classes within the bag could also be informative for predicting the bag label. 
  - We extend our model to learn relationships between classes by adding an additional fully connected layer following the
MIL pooling layer. 
  -  We formulate a joint cross entropy objective function at both the MIL pooling layer and the additional fully connected layer defined by
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/11b81e65-4cea-417e-b846-c30db22c8737' width=40%>


#### Global pooling functions
- We explore the use of several different global pooling functions $g(\cdot)$ in our model.
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/570f570f-ab6f-4af5-8fac-ec990c7dbbad' width=40%>


#### Localizing cells with Jacobian maps
-  We aimed to extend our model by localizing regions of the full resolution input images that are responsible for activating the class specific feature maps.
- We employ recently developed methods for visualizing network activations toward this purpose. 
- Our approach is similar to Simonyan et al. (2013) in which the pre-softmax activations of specific output nodes are back propagated through a classification network to generate Jacobian maps w.r.t. specific class predictions. 
  - Let $a^{(l)}$ be the hidden activations in layer l and $z^{(l)}$ be pre-nonlinearity activations. 
  - We define a general recursive non-linear back-propagation process computing a backward activation a for each layer,
analogous to the forward propagation:
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/d14fd9e9-1213-4204-8b91-7d1506e31c0c' width=40%>

  - In our case, we start the non-linear back-propagation $(a_{ij}^L)$ from the MIL layer using its sigmoidal activations for the class i specific feature maps $(p_{ij})$ multiplied by the pooling activation for each class $P_i \cdot p_{ij}$.
  - We find that applying the ReLU activation function to the partial derivatives during back propagation generates Jacobian maps that are sharper and more localized to relevant objects in the input. To generate segmentation masks we threshold the sum of the Jacobian maps along the input channels. 

### Dataset: Yeast protein localization screen
- We used a genome wide screen of protein localization in yeast containing images of 4144 yeast strains from the yeast
GFP collection  representing 71% of the yeast proteome.
- We categorized whole images of strains into 17 localization classes based on visually assigned localization annotations from a previous screen.
-  These labels include proteins that were annotated to localize to more than one sub-cellular compartment. 
  - We evaluated all the proteins in the screen and report the test error for the 998 proteins that are localized to a single compartment and mean average precision for the 2592 proteins analyzed in


### Model Training
- We extracted slightly smaller crops of the original images to account for variability in image sizes within the screens (we used 1000  1200 for the breast cancer dataset and 1000  1300 for the yeast dataset). 
- We normalized the images by subtracting the mean and dividing by the standard deviation of each channel in our training sets. 
- During training we cropped random 900  900 patches from the full resolution images and applied random rotations and reflections to the patches.

### Results
- Huh indicates agreement with manually assigned protein localizations
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/4624ef2a-9c5b-40c5-b7e5-03e81eb23c0d' width=50%>