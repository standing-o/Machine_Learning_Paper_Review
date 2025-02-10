## Advancing medical imaging informatics by deep learning-based domain adaptation
- Authors: Choudhary, Anirudh and Tong, Li and Zhu, Yuanda and Wang, May D
- Journal: Yearbook of medical informatics
- Year: 2020
- Link: https://www.thieme-connect.com/products/ejournals/pdf/10.1055/s-0040-1702009.pdf


### Summary
- Getting largescale labeled data remains a challenge, and multi-center datasets suffer from heterogeneity due to patient diversity and varying imaging protocols. 
- `Domain adaptation (DA)` has been developed to transfer the knowledge from a labeled data domain to a related but unlabeled domain in either image space or feature space.
- `DA` is a type of transfer learning (TL) that can improve the performance of models when applied to multiple different datasets. 
- We discussed domain transformation (DT) and latent feature-space transformation (LFST).


### Introduction
- While multicenter datasets can increase the amount of annotated data, these datasets suffer from heterogeneity due to varying
hospital procedures and diverse patient populations.
  - Due to a distribution shift (also known as domain-shift) between the available training dataset and the dataset encountered in clinical practice, pre-trained models trained by one dataset may fail for another dataset. 

#### What is Transfer Learning and `Domain Adaptation`
- Transfer learning (TL) applies knowledge learned from one domain and one task to another related domain and/or another task.
- For medical imaging, a domain usually refers to images or features, while the task refers to segmentation, classification, etc.
- If both source $(D_s)$ and target domains $(D_T)$ are similar, then $D_s$ and $D_T$ can use the same ML model for similar tasks $(T_S ~ T_T)$.
  - If $D_s \neq D_T$ or $T_S \neq T_T$, the ML model trained on the source domain might habve decreased performance on the target domain $(D_T)$.
  1. Inductive TL requires some labeled data. While the two domains may or may not differ $(D_s ~ D_T)$ or $(D_s \neq D_T)$, the target and source tasks are different $(T_S \neq T_T)$.
  ➔ ex. Lung tumor detection across X-Ray and computed tomography images.
  2. Transductive TL requires labeled source data and unlabeled target data with related domains $(D_s ~ D_T)$ and same tasks
$(D_s = D_T)$, while the marginal probability distributions differ $(p(X_S \neq X_T))$.
  ➔ ex. Lung tumor detection across X-Ray and computed tomography images.
  3. Unsupervised TL does not require labeled data in any domain and has different tasks $(T_S \neq T_T)$.
  ➔ ex. Classifying cancer for different anatomies using unlabeled histology images.
- `Domain Adaptation (DA)` is a **2. transductive TL** approach that aims to transfer knowledge across domains by learning domain-invariant transformations, which align the domain distributions.

    <img src="https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/a1f0a0c5-b95c-41ee-a32a-f18143a40627" width=90%>


#### Using Domain Adaptation to improve model training in medical imaging
- Cross-modal `DA` transfers labels between distinct, but somewhat related, image modalities.
  - Single-modality `DA` adapts different image distributions within the same modality.
- `DA` can mitigate the lack of well-annotated data by augmenting target domain data, either by generating synthetic labeled images from source images or aligning source and target image features and training a task network on them.
- Through `DA`, annotated MRI scans from historical subjects can be combined with CT to reduce the number of image acquisitions needed.
- H&E stained images are widely available, while IHC images, which clearly highlight nuclei via specific biomarkers, are not.
  - `DA` methods can translate multi-stained H&E-stained images to the IHC domain, making nuclei detection easier.
- `Domain Adaptation` methodologies
    <img src="https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/6a2433a8-fb10-4210-ba40-768eef9beea2" width=90%>


#### Challenge of Dataset Variations
- Pathology images have stain variations whiel MRIs are susceptible to varing magnetic fields and contrast agents.
- Such intra- or inter-dataset variations cause the tarining and test dataaset to have different distributions, resulting in a domian-shift which impacts model generalization.
- `DA` methods try to minimize the dataset variation, while retaining the distinguishing aspects for task classifier, and have been shown to generalize well in image sesgmentation tasks for multiple modalities.

### Deep Learning-based `Domain Adaptation`
- Two families of DA approaches for medical imaging: DT-DA and LSFT-DA

#### Domain transformation in `domain adaptation` (DT-DA)
- DT-DA translates images from one domain to the other domain, so that the obtained models can be directly applied to all images
  - Such translation is typically done using generative models, which achieve pixel-level mapping by learning the translation at a semantic level.
  - The translation direction is usually decided by the relative ease of translation and modeling in a modality.
- DT-DA performs alignment in the image space instead of the latent feature space, leading to better interpretability through visual inspection of synthesized images, enforcing semantic consistency, and preserving low-level appearance aspects using shape-consistency and structural-similarity constraints.
- **Undirectional Translation** 
  - Undirectional translation maps images from the source domain to the target domain or vice versa using GANs.
  - Undirectional translation has been applied to remove dataset variations.
    - Bentaieb et al. designed a stain normalization approach, using a task conditional GAN to translate H&E images to a reference stain.

- **Bidirectional Translation**
  - Bidirectional image translation (reconstruction-based DT) leverages two GANs, constraining the mapping space by enforcing semantic-consistency between toe original and reconstructed images.
  - CycleGAN has been expanded to handle larger domain shifts with semantic-consistency loss functions, multi-domain translation, and translation between two domains with multi-modal conditional distributions.
  - Bidirectional translation expands the training data to make the segmentation task model robust.
    - The translation and segmentation network can be trained either independently or jointly.

#### Latent feature space transformation in `domain adaptation` (LFST-DA)
- LFST-DA transforms the source domain and target domain images to a shared latent feature space to learn a domain-in-variant feature representation.
  - The goal is to minimize domain-specific information while preserving the task-related information.
- LFST-DA is more computationally efficient because it focuses on translating relevant information only instead of the complete image.
- LFST-DA is used in three basic implementations:
  - **Divergence minimization**
    - A simple approach to learn domain-invariant features and remove distributions-shift is to minimize some divergence criterion between source ans target data distributions.
    - ex. maximum mean discrepancy, correlation alignment, contrastive domain discrepancy, wasserstein distance.
  - **Adversarial Training**
    - Adversarial methods train a discriminator, typically a seperate network, in an adversarial fashion against the feature encoder network.
    - The goal of the feature network is to learn a latent representation s.t. the discriminator is unable to identify the input sample domain from the representation.
    - Zhang at el. applied a domain discriminator to adapt models trained for pathology images to microscopy images.
    - LSFT-DA is also used for single-modality adaptation to overcome dataset variations in pathology images.
  - **Reconstruction-based adaptation**
    - The reconstruction-based adaptation maximizes the inter-domain similarity by encoding images from each domain to reconstruct images in the other domain.
    - The reconstuction network (decoder) performs feature alignment by recreating the feature extractor's input while the feature extractor (encoder) transforms input image into latent representation.
    - Bousmalis et al. proposed a domain separation network that extracts image representations in two subspaces: the private domain features and the shared-domain features, the latter being used to reconstruct input image.


### Challenges and Opportunities
#### Domain selection and direction of `domain adaptation`
- In medical imaging, domains are often selected based on the type of imaging techniques, anatomy, availability of labeled data, and whether the modalities are complementary for the underlying task.
- Regarding whether `DA` could be performed symmetrically across domains, the potential information loss in a particular direction is critical for assessing task performance.
- To assess domain relationship and `DA` direction, it is necessary to use:
  - Large-scale empirical studies such as exploring bi-directional `DA` across multiple datasets
  - Representation-shift metric to roughly quantify the risk of applying learned-representations from a particular domain to new domain
  - Multi-source `DA` automatically explores latent source domains in multi source datasets and quantifies the membership of each target sample.

#### Transferability of individual samples
- Most `DA` studies for medical imaging assume that all samples are equally transferable across two domains.
- They focus on globally aligning domain distributions.
- The ability to transfer varies across clinical samples because of:
  - Intra-domain variations, noisy annotations due to human subjectivity, target label space being a subset of source label space, varing transferability among different image regions.

#### Limitations of `domain adaptation` in medical imaging
- Adversarial methods are prone to errors because the discriminator can be confused, and there is no gurantee that the domain distributions are sufficiently similar.
- The generator in GAN is prone to hallucinating content to convince the discriminator that data belongs to the target distribution.


### Conclusions and Future Directions
- `DA` has emerged as an effective approach for minimizing domain-shift and leveraging labeled data from distinct but related domains.
- LSFT-DA and DT-DA are two popular approaches to minimize the distribution divergence in multiple medical imaging studies exploring same-modality or cross-modality scenarios.