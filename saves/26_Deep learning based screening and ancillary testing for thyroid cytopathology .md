## Deep learning based screening and ancillary testing for thyroid cytopathology
- Authors: Dov, David and Range et al.
- Journal: The American journal of pathology
- Year: 2023
- Link: [https://pdf.sciencedirectassets.com...](https://pdf.sciencedirectassets.com/280296/1-s2.0-S0002944022X00102/1-s2.0-S0002944023002031/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEDsaCXVzLWVhc3QtMSJHMEUCICpwCYSLb37MIJMq%2FZwwNb2O9AZVMXcAZj9QalurFtsMAiEA%2BKpVMU6drmQZ8MHt%2BkY590h6Olo8r7a3i7c2tffEONUqvAUI5P%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDGs%2BUBMkqEHPzfgDkiqQBXoMsdfoWnakEmFpqFW4GR6S7YEAAh3idoiz3Az3PIQisM666PwDxPeJDWvbveIUJkYQbdIGtXOTOJwHZtdJ%2Fk7j8cIUxU3dDJiQkCSFwmMqbjMl285beqV7e15z8fWsVBjWg43fVUCQ66oeM2hVuH8ySNRrxMw%2BuL%2FNh1G6%2BqAaIxBatA3jK5sq8XQFLhzgmFMvRfMGNB0hq4DbWJbwCweX12Al6mBw%2FyaRsYGJX5Dvmfgr153i02Pbt%2FBmzmVYE0xpD55mWOXo2Oja2Ejb7MPhrley9mXa1bklGyx2xMYeGRx5TPD0Ukaab2iRfkPewj8yyZwQdETNC%2BX%2FgF5zYINP%2Bo8xicGRqGeZ09Nyv8qfL%2F5dkF3AUeS4MtUq81yAMQbjQ2m2hEVy2FFLChqWTyjGNGKDqr4YGi8EnBkU61nbpQGyYSPpJgRCH2eGHBeTrISTRSOVCQK4hFK5ydclhS3izNHnjGvDeIx52%2BBAigElKEpxE24Kmbth6zQMuEYgqscRP2ufWIjAXAidOflGBKLZqa6ksmUrAongwHw%2BAypsh8cEi3fCUUW9BmVtK46tVhEp2eI4GL2Sy3%2BN0FAKsQVor5l5mrZL%2FwkpE1ZBkJALM3baBW7sRtRu62Kn1PEPtps7QA6i2SikFokoVknDspk44Ai2tJuuRDgZL%2B0pFqHPc%2FyiC3qFhme2bp7KVfYXgsliB2PATJ%2FB6MmG4fFCqZ4ALI3c14pynl1bM2hFdQKZEOOhFvzr5HIWiBtvNDgu%2F1VVO9OjgLPXwLR5a%2FFkhnAnDKGvOwpeddzJIvSTe5sTm6CxKTxUILaT4Rz99rEyDoEuaTN%2Bb73J0GcRIXaV%2FegFqRwy2g04yOu9opkmQS2OMJqk87MGOrEBqLt2zDy0fXmK0cDY2M5eLBEImjxE%2BnvZtL1ZZpNKXj1laGR3fFxQaulUgATdAEqbiWHN05nGHl%2FEo%2Fdll6Hueb2eqTn7Vj1EXMynaDJSeLPCm3DgmnTKvK75k4wO5b9XgHmiOfr0BvrFFeqSMohv7RVjmggJf1i%2Bi2e%2BxbT%2BK%2FsNDeBFtuoUtu0zYoAYswV1hvjbAX8%2FWO5q1FPR07z%2B3Mq50VlyFVk1x0QtEGpKbX83&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240627T031059Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYWEUQUL6C%2F20240627%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=f0513cc2b942ecfa05a1dfe88980eb3d50b0d6758aefaf0a9ab9279e09968d59&hash=33027c0b02eeb11eb72d1ea1f6728a7aa09630983f692bd73bbb751c0bb898d8&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0002944023002031&tid=spdf-66f6fd40-80e0-4e1a-b67d-0333adfda1c5&sid=9060f63b2c311344c13a140506a5e78b6c82gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=09145c5e06545c0151&rr=89a23f42390a3290&cc=kr)





### Introduction
- In this article, we propose a deep-learninge based ternary, rather than a classic binary, model, which classifies each
FNAB scan into one of the three categories: _benign, indeterminate, or malignant. 
- At the expense of providing indeterminate classifications in some of the cases, the model is tuned to provide accurate low and high ROM (Risk of Malignancy) for the benign and malignant categories, respectively. 
- This approach allows us to apply the algorithm in two practical use cases while achieving clinical-grade performance:
  - Screening to identify determinate cases (i.e., providing definitive and reliable predictions that do not require further manual review by pathologists)
  - Ancillary testing for disambiguating and reducing the number of indeterminate cases, to help reduce unnecessary surgeries. The algorithm screens and definitively classifies 45.1% (130/288) of the scans as either benign or malignant, while providing human expertlevel ROMs of 2.7% and 94.7%, respectively. The algorithm further reduces the number of indeterminate cases by
definitively classifying 21.3% (23/108) with a ROM of 1.8%.

### Dataset
- The cohort comprised 2169 FNAB slides. The authors excluded FNABs diagnosed as non-diagnostic by the EMR
cytopathologist (CP), which comprised 3.2% of the cases. The final data set was divided into a training set of 964 FNABs and a test set of 601 FNABs. 
- All slides were cleaned and scanned with a 40 objective and nine levels of Z-stack on a Leica AT-2 scanner. 
- The authors used the middle Z-stack, which was further down-sampled by a factor of four in each dimension to reduce processing time.

### Algorithm
- The proposed algorithm is inspired by the workflow of cytopathologists and comprises two CNNs - `Informativeness CNN`, `Malignancy CNN`. 
  - The first network, termed `informativeness CNN`, discriminates thyroid follicular cells. These diagnostically relevant areas
typically comprise only a tiny fraction of the entire slide, which is otherwise mostly occupied by irrelevant cellular material (eg, blood cells). The `informativeness CNN` mitigates this challenge by selecting only relevant areas, effectively reducing data dimensionality.
  - The second network, malignancy CNN, classifies FNABs into the three clinically relevant categories (benign, indeterminate, or malignant). The classification is based on ordinal regression whereby a scalar output of the network is compared with two learnable threshold parameters. 
    - During the training phase, the threshold parameters are tuned together with the parameters of the neural network via
stochastic gradient descent. By the nature of ordinal regression, the three categories reflect increasing probability of malignancy.
- Both CNNs are based on the widely used VGG11 pretrained on Imagenet.
- Each of the RGB color channels of the scans was normalized to have 0 mean and variance 1. Then, the scans were tiled into patches of 128 x 128 pixels and fed into the `informativeness CNN` that predicts if they are informative (i.e., contain thyroid follicular cells). 
- During the training, the informative patches were sampled from the subset of regions manually marked by D.E.R. in the training
set. 
  - Direct smears made from FNABs contain far more uninformative regions (blank regions, blood cells, and artifacts) than informative ones. In some scans, which typically contain hundreds of thousands of patches, merely a few of them are informative. Because manually annotating regions in the scans is extremely time-consuming, the authors decided to devote this effort to only mark informative regions. The uninformative regions were sampled uniformly from the WSI given the overwhelmingly high likelihood of
sampling background/negative areas. 
  - After completing the training process, a sliding window sweeps over the WSI, and the CNN predicts the informativeness of each patch. For each WSI, the most informative patches are selected and organized into a set of patches of a fixed size, which are then fed into the malignancy CNN. The authors used 1000 patches per WSI for training the malignancy CNN, a number that provides a sufficiently large amount of data to train the neural network.
  - The authors found this scheme efficient in extracting the informative regions, while filtering out white space and irrelevant material. The authors’ patches selection strategy allowed selecting overlapping patches. Therefore, when the number of informative regions was smaller than the fixed number of selected patches (1000 in training and 100 in testing), the `informativeness CNN` usually selected overlapping regions.
  - An alternative approach is selecting only patches with prediction value of the `informativeness CNN` higher than a certain threshold value. However, there is no straightforward way to select the threshold value, and this alternative approach did not provide improvement in early experiments.
- The `malignancy CNN` provides predictions of the final surgical pathology diagnosis by averaging the predictions obtained from each patch in the set. To transform the algorithm’s predictions into clinically relevant classifications of benign, indeterminate, and malignant categories, the authors used learnable threshold parameters to which they compare the (continuous) output of the `malignancy CNN`.   
  - Let $p \in [0, 1]$ be the (continuous) output of the `malignancy CNN` (after the sigmoid layer) and let $\tau_1, \tau_2, \tau_3, \tau_4 \in \mathbb{R}$ be the learnable thresholds. The thresholds divide the predictions into ranges associated with the different TBS categories, each with an increased risk of malignancy.
  - This strategy allows the authors to automatically tune the threshold parameters as part of the training process while allowing the
`malignancy CNN` to learn from the final pathology labels, which are the gold standard/ground truth. During testing, the

<img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/29dda165-1d2b-44be-9b89-88274ba830aa'>

### Results
- As a secondary question, the authors wondered how the algorithm performed among cases for which the CPs showed some disagreement. These are presumably more difficult cases, which are less likely to be in the determinate categories of benign or malignant.

<img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/ac733f03-fb38-414a-a5e7-fce622781e4e'>


