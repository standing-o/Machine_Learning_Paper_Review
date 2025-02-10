## Why do tree-based models still outperform deep learning on typical tabular data?
- Authors : Grinsztajn, L{\'e}o and Oyallon, Edouard and Varoquaux, Ga{\"e}l
- Journal : Advances in Neural Information Processing Systems
- Year : 2022
- Link : https://proceedings.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf

### Abstract
- The superiority of deep learning on tabular data is not clear.
- Tree-based models remain state-of-the-art on medium-sized data (~10K samples) even without accounting for their superior speed.     
➔ To understand this gap, we conduct an empirical investigation into the differing inductive biases of tree-based models and neural networks.
- Guide to build tabular-specific neural network:    
1. Be robust to uninformative features
2. Preserve the orientation of the data
3. Be able to easily learn irregular functions

### Introduction
- Deep learning architecture have been crafted to create inductive biases matching invariances and spatial dependencies of the data. Finding corresponding invariances is hard in tabular data, made of heterogeneous features, small sample sizes, extreme values.
- One motivation is that tree-based models are not differentiable,  and thus cannot be easily composed and jointly trained with other deep learning blocks.

- Most tabular datasets available are small.
- Tree-based models remain state-of-the-art on medium-sized tabular datasets, even without accounting for the slower training of deep learning algorithms.
- Understand which inductive biases make them well-suited for these data. By transforming tabular datasets to modify
the performances of different models, we uncover differing biases of tree-based models and deep learning algorithms which partly explain their different performances: neural networks struggle to learn irregular patterns of the target function, and their rotation invariance hurt their performance, in particular when handling the numerous uninformative features present in tabular data.

### A benchmark for tabular learning
- **Heterogeneous columns**: Each column corresponds to the same signal on different sensors.
- **Not high dimensiona**l: Keep datasets with a d/n ratio below 1/10, and with d below 500.
- **Undocumented datasets**: Remove datasets where too little information is available
- **I.I.D data**: Remove stream-like datasets or time series.
- **Real-world data**: Remove artificial datasets but keep some simulated datasets.
- **Not too small**: Remove datasets with too few features and too few samples.
- **Not too easy**: Remove datasets which are too easy.
- **Not deterministic**: Remove datasets where the target is a deterministic function of the data.
- **Medium-sized training dataset**: Truncate the training set to 10,000 samples for bigger datasets.
- **No missing data**: Remove all missing data from the datasets.
- **Balanced classes**
- **Low cardinality categorical features**: Remove categorical features with more than 20 items.
- **High cardinality numerical features**: Remove numerical features with less than 10 unique values.

### A procedure to benchmark models with hyper-parameter selection
- Hyperparameter tuning leads to uncontrolled variance on a benchmark especially with a small budget of model evaluations.
- To study performance as a function of the number n of random search iterations, we compute the best hyper-parameter combination on the validation set on these n iterations.

### Aggregating results across datasets
- We use the test set accuracy and R2 score to measure model performance and a metric similar to the distance to the minimum.

### Data preparation (Pre-processing)
- Gaussinized features, Transformed regression targets, OneHotEncoder

## Tree-based models still outperform deep learning on tabular data
- Tree-based models benchmarked: Random Forest, Gradient Boosting Trees, XGBoost
- Deep models benchmarked: MLP, Resnet, FT_Transformer
- Benchmark on medium-sized datasets
<img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/9ec6d049-4e5f-4234-8cfd-98d7adf752ae' width=60%>

### Results
- Tuning hyper-parameters does not make neural networks state-of-the-art    
➔ Tree-based models are superior for every random search budgets.
- Categorical variables are note the main weakness of neural networks    
➔ Most of the gap subsists when learning on numerical features only.

## Empirical investigation: why do tree-based models still outperform deep learning on tabular data?
### Methodology: uncovering inductive biases  
- Inherent properties of the models explain their performances on tabular data      
➔ The best methods on tabular data share: ensemble methods, decision tree.
- We apply various transformations to tabular datasets which either narrow or widen the generalization performance gap between neural networks and tree-based models     
➔ emphasize their different inductive biases.

### Finding 1: Neural networks are biased to overly smooth solutions
- We transform each train set by smoothing the output with a Gaussian Kernel smoother for varying length-scale values of the kernel.
- Performance as a function of the length-scale of the smoothing kernel.
➔ The target functions in our datasets are not smooth, and that neural networks struggle to fit these irregular functions compared to tree-based models. 
<img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/f99841f4-a3c6-42f2-b520-f7fdc08fb7be' width=60%>


### Finding 2: Uninformative features affect more MLP-like neural networks
- Tabular datasets contain many uninformative features     
➔ The classification accuracy of a GBT is not much affected by removing up to half of the features.
<img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/3579144c-1dbb-4bd8-aa51-111ca09bd828' width=60%>

- MLP-like architectures are not robust to uninformative features
➔ Removing uninformative features reduces the performance gap between MLPs an the other models, while adding uninformative features widens the gap.
<img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/eac8c450-3795-4219-ad00-7a876b707eb5' width=60%>

### Finding 3: Data are non invariant by rotation, so should be learning procedures
- The learning procedure which learns an MLP on a training set and evaluate it on a testing set is uncahnged when applying a rotation to the features on both the training and testing set.
- Tree-based models are not rotationally invariant, as they attend to each feature separately.    
➔ Neither are FT Transformer: Initial FT Tokenizer implements a pointwise operation.
- Intuitively, to remove uninformative features, a rotationaly invariant algorithm has to first find the original orientation of the features, and then select the least informative ones: the information contained in the orientation of the data is lost.
-  The change in test accuracy when randomly rotating our datasets, confirms that only Resnets are rotationally invariant.    
➔ Random rotations reverse the performance order:  rotation invariance is not desirable
<img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/1a9a0b85-8886-4a29-8e46-51ad3a330e63' width=40%>

-  Removing the least important half of the features in each dataset (before rotating), drops the performance of all models except Resnets, but the decrease is less significant than when using all features.
<img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/1cbbcbe0-de74-4a14-9cc4-22374878f3e4' width=40%>

## Conclusion
- Tree-based models more easily yield good predictions, with much less computational cost. This superiority is explained by specific features of tabular data: irregular patterns in the target function, uninformative features, and non rotationally-invariant data where linear combinations of features misrepresent the information. 