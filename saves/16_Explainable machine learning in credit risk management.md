## Explainable machine learning in credit risk management
- Authors : Bussmann, Niklas and Giudici, Paolo and Marinelli, Dimitri and Papenbrock, Jochen
- Journal : Computational Economics
- Year : 2021
- Link : http://text2fa.ir/wp-content/uploads/Text2fa.ir-Explainable-Machine-Learning-in-Credit.pdf

### Abstract
- The model applies correlation networks to Shapley values so that Artifcial Intelligence predictions are grouped according to the similarity in the underlying explanations.

### Introduction
- Black box Artifcial Intelligence (AI) is not suitable in regulated fnancial services. 
➔ Explainable AI models provide details or reasons to make the functioning of AI clear or easy to understand.
- Simple statistical learning models, such as linear and logistic regression models, provide a high interpretability
but, possibly, a limited predictive accuracy.
- Complex machine learning models, such as neural networks and tree models, provide a high predictive accuracy at the expense of a limited interpretability
➔ Our proposed methodology acts in the post processing phase of the analysis, rather than in the preprocessing part. It is agnostic (technologically neutral) as it is applied to the predictive output, regardless of which model generated it: a linear regression, a classifcation tree or a neural network model.
- It utilises a model-agnostic method aiming at identifying the decision-making criteria of an AI system in the form of variable importance.
➔ `Shapley` value decomposition of a model, a pay-of concept from cooperative game theory.   
- Our proposed methodology : the combination of network analysis with `Shapley` values (Shapley, 1953)
  - `Shapley` values correspond to the average of the marginal contributions of the players associated with all their possible orders. 
  - The advantage of `Shapley` values, over alternative XAI models, is that they can be exploited to measure the contribution of each explanatory variable for each point prediction of a machine learning model, regardless of the underlying model itself.
  - Our original contribution is to improve `Shapley` values, improving the interpretation of the predictive output of a machine learning model by means of correlation network models.
  ➔ Correlation networks, also known as similarity networks, have been introduced by  Mantegna and Stanley (1999) to show how time series of asset prices can be clustered in groups on the basis of their correlation matrix.

### Methodology
#### Machine Learning of Credit Risk
- In highly regulated sectors, like fnance or medicine, models should be chosen balancing accuracy with explainability.
➔ We consider the Extreme Gradient Boosting (`XGBoost`), which is a supervised model based on the combination of tree models with Gradient Boosting.
- In practice, a tree classifcation algorithm is applied successively to “training” samples of the data set. In each iteration, a sample of observations is drawn from the available data, using sampling weights which change over time, weighting more the observations with the worst ft. 
➔ Once a sequence of trees is ft, and classifcations
made, a weighted majority vote is taken.

#### Learning model comparison
- Build the model using data in the train set, and compare the predictions the model obtains on the test set, Ŷ<sub>n</sub>, with the actual values of Y<sub>n</sub>.
- To obtain Ŷ<sub>n</sub> the estimated default probability is rounded into a “default” or “non default”, depending on whether a threshold is passed or not. 
- For a given threshold T, one can then count the frequency of the four possible outputs, namely: FP,TP,FN,TN
- Receiver Operating Characteristics (ROC) curve, which plots the false positive rate (FPR) on the Y axis against the true positive rate (TPR) on the X axis, for a range of threshold values (usually percentile values).

#### Explaining model predictions
- We now explain how to exploit the information contained in the explanatory variables to localise and cluster the position of each individual (company) in the sample.
- We develop our `Shapley` approach using the `SHAP` (Lundberg and Lee, 2017) computational framework, which allows to estimate `Shapley` values expressing predictions as linear combinations of binary variables that describe whether each single variable is included or not in the model.
- Once `Shapley` values are calculated, we propose to employ similarity networks, defning a metric that provides the relative distance between companies by applying the Euclidean distance between each pair of company predicted vectors (Giudici et al, 2019).
- We then derive the Minimal Spanning Tree (`MST`) (Mantegna and Stanley, 1999) representation of the companies, employing the correlation network method.
➔ Reason for choice of `MST` :  `MST` simplifes the graph into a tree of N−1 edges, which takes N−1 steps to be completed. At each step, it joins the two companies that are closest, in terms of the Euclidean distance.
➔ `MST`-`Shapley` : In our Shapley value context, the similarity of variable contributions is expressed
as a symmetric matrix of dimension n × n, where n Is the number of train data points. The `MST` representation associates
to each point its closest neighbour.

### Data
- European External Credit Assess‑ment Institution (ECAI) that specializes in credit scoring for P2P platforms focused on SME commercial lending. The information about the status (0 = active, 1 = defaulted) of each company one year later (2016) is also provided.
- XGBoost classification, Similarity network in a post-preprocessing step, Cluster dendrogram representation (`MST`)

### Result
- XGBoost training ➔ `Shapley` value explanation (using `MST`)
- AUROC for two models:
<img src='https://user-images.githubusercontent.com/57218700/181426054-bcd411b7-30ad-4fba-9b27-c9751d46cd9c.png' width=40%>
➔ XGBoost improves predictive accuracy.

- Nodes of `MST` are colored according to their cluster of belonging.
<img src='https://user-images.githubusercontent.com/57218700/181425328-1e176716-151e-4830-8142-c2dee5ef316c.png' width=40%>
➔ Clusters are quite scattered along the correlation network.

- Company nodes are colored according to their status : not defaulted (grey), defaulted (red).
<img src='https://user-images.githubusercontent.com/57218700/181425656-91f071a1-ebef-4936-8c05-fa4356ae1bf5.png' width=40%>   
➔ Default nodes appear grouped together in the `MST`, particularly along the bottom left branch. In general, defaulted institutions occupy precise portion of the network, usually to the leafs of the tree, and form clusters.    
➔ Those companies form communities, characterised by similar predictor variables’ importances.    
➔ Not defaulted companies that are close to default ones have a high risk of becoming defaulted as well, being the importance of their predictor variables very similar to those of the defaulted companies.   

- Contribution of each explanatory variable to the `Shapley`’s decomposition of four predicted default probabilities, for two defaulted and two non defaulted companies.
<img src='https://user-images.githubusercontent.com/57218700/181425953-9d552b80-0a40-406e-9f7b-a2f1bb7727e9.png' width=60%>
➔ The most important variables, for the two non defaulted companies (left boxes) regard: profts before taxes plus interests paid, and earnings before income tax and depreciation (EBITDA), which are common to both; trade receivables, for company 1; total assets, for company 2.

- Mean contribution of each explanatory variable to the `Shapley`’s decomposition. The more red the color the higher the negative importance, and the more blue the color the higher the positive importance.
<img src='https://user-images.githubusercontent.com/57218700/181426453-bda986e5-4f99-4e13-b3f9-14c5d29f2061.png' width=60%>
➔ Total assets to total liabilities (the leverage) is the most important variable, followed by the EBITDA, along with proft before taxes
plus interest paid, measures of operational efciency; and by trade receivables, related to solvency.

### Conclusion and Future research
- The model can explain, from a substantial viewpoint, any single prediction in terms of the `Shapley` value contribution of each explanatory variables.
- Future research should extend the proposed methodology to other datasets and, in particular, to imbalanced ones, for which the occurrence of defaults tends to be rare, even more than what observed for the analysed data.