# BioMapAI
The primary goal of BioMapAI is to connect high-dimensional biology data, $X$ to mixed-type output matrix, $Y$. Unlike traditional ML or DL classifiers that typically predict a single outcome, $y$, BioMapAI is designed to learn multiple objects, $Y=\left[y_1,\ y_2,\ \ldots,y_n\right]$, simultaneously within a single model. This approach allows for the simultaneous prediction of diverse clinical outcomes - including binary, categorical, continuous variables - with ‘omics profiles, thus address disease heterogeneity by tailoring each patient’s specific symptomatology.

## To Train your BioMapAI

BioMapAI is a deep learning framework for multi-stage modeling of biological data. It first predicts intermediate targets (omics scores) and then maps them into a final outcome or classification label.

## Features

- **OmicScoreModel**: Train a model to predict intermediate omic scores (Y).  
- **ScoreLayer**: Build a simple layer (or sub-model) that converts omic scores (Y) into the final target (y0).  
- **ScoreYModel**: Combine the trained omic score model with the ScoreLayer to get final predictions and metrics.  
- **WeightsAdjust** (Optional): Fine-tune the relationship between Y and y0 for better performance.

## Quick Start

1. **Install Dependencies**  
   ```bash
   pip install numpy pandas tensorflow```

2. **Clone or Download This Repository**

   This repository should include:
   - `DNN.py`: Contains the classes and methods (OmicScoreModel, ScoreLayer, etc.).
   - `example_data/`: Folder containing `train_data.csv` and `test_data.csv`.
   - `OmicScoreModel_Tutorial.ipynb`: Detailed notebook tutorial.

3. **Run the Tutorial Notebook**

   - Open `OmicScoreModel_Tutorial.ipynb` in Jupyter Notebook or JupyterLab.
   - Follow the cells step-by-step to:
     1. Load training and test data.
     2. Train the **OmicScoreModel** to predict intermediate scores (`Y`).
     3. Build a **ScoreLayer** to convert those scores into final predictions (`y0`).
     4. Evaluate the model performance on a test set.
     5. (Optional) Adjust weights to improve performance.

4. **Customize or Extend**

   - Tune hyperparameters (epochs, optimizer, batch size, etc.).
   - Add or remove features in the data CSV files.
   - Modify `DNN.py` to create custom network architectures or loss functions.
   - Integrate advanced data preprocessing or feature engineering techniques.

## Further Details

For an in-depth guide, check out the [OmicScoreModel_Tutorial.ipynb](OmicScoreModel_Tutorial.ipynb). It covers:

- Data loading and organization  
- Model instantiation and training procedures  
- How to evaluate intermediate and final predictions  
- Strategies for adjusting the model to improve performance  

---


## To Load and use our pretrained model for ME/CFS: DeepMECFS




## 1. BioMapAI Structure.
BioMapAI is a fully connected deep neural network framework comprising an input layer $X$, a normalization layer, three sequential hidden layers, $Z^1,\ Z^2,\ Z^3$,and one output layer $Y$.

1) **Input layer ($X$)** takes high-dimensional ‘omics data, such as gene expression, species abundance, metabolome matrix, or any customized matrix like immune profiling and blood labs.
2) **Normalization Layer** standardizes the input features to have zero mean and unit variance, defined as

   $X^\prime=\frac{X-\mu}{\sigma}$
   
where $\mu$ is the mean and $\sigma$ is the standard deviation of the input features.
3) **Feature Learning Module** is the core of BioMapAI, responsible for extracting and learning important patterns from input data. Each fully connected layer (hidden layer is designed to capture complex interactions between features. **Hidden Layer 1 ($Z^1$)** and **Hidden Layer 2 ($Z^2$)** contain 64 and 32 nodes, respectively, both with ReLU activation and a 50% dropout rate, defined as:

$Z^k=ReLU\left(W^kZ^{k-1}+b^k\right),\ \ k\in{1,2}$

**Hidden Layer 3 (\mathbit{Z}^\mathbf{3})** has n parallel sub-layers for each object, $y_i$ in $Y$. Every sub-layer, $Z_i^3$, contains 8 nodes, represented as:

$Z_i^3=ReLU\left(W_i^3Z^3+b_i^3\right),\ \ i\in{1,2,\ldots,n}$

All hidden layers used ReLU activation functions, defined as:

$\mathrm{ReLU}\left(x\right)=max\left(0,x\right)$

4) **Outcome Prediction Module** is responsible for the final prediction of the objects. The output layer ($Y$) has n nodes, each representing a different object:

$y_i\ =\ σ(Wi4Zi3+bi4)                          for binary objectsoftmax(Wi4Zi3+bi4)      for categorical object Wi4Zi3+bi4                         for continuous object$

The loss functions are dynamically assigned based on the type of each object: 

$\mathcal{L}=\ 1Ni=1Nyilogyi+1-yilog1-yi       for binary object-1Ni=1Nj=1Cyijlogyij                          for categorical object1Ni=1N0.5yi-yi2,   if yi-yi≤δδyi-yi-0.5δ2,otherwise      for continuous object$  

During training, the weights are adjusted using the Adam optimizer. The learning rate was set to 0.01, and weights were initialized using the He normal initializer. L2 regularizations were applied to prevent overfitting.

5) **Optional Binary Classification Layer** (not used for parameter training). An additional binary classification layer is attached to the output layer Y to evaluate the model's performance in binary classification tasks. This layer is not used for training BioMapAI but serves as an auxiliary component to assess the accuracy of predicting binary outcomes, for example, disease vs. control. This ScoreLayer takes the predicted scores from the output layer and performs binary classification:

$y_{binary}=\sigma\left(W_{binary}Y+b_{binary}\right)$

The initial weights of the 12 scores are derived from the original clinical data, and the weights are adjusted based on the accuracy of BioMapAI's predictions:

$w_{\mathrm{new}}=w_{\mathrm{old}}-\eta\nabla\mathcal{L}_{MSE}$

where $\nabla\mathcal{L}_{MSE}$ refers to the mean squared error (MSE) between the predicted $y’$ and true $y$, then adjusts the weights to optimize the accuracy of the binary classification.


## 2. Training and Evaluation of BioMapAI for ME/CFS – BioMapAI::DeepMECFS. 
BioMapAI is a framework designed to connect high-dimensional, sparse biological ‘omics matrix X to multi-output Y. While BioMapAI is not tailored to a specific disease, it is versatile and applicable to a broad range of biomedical topics. In this study, we trained and validated BioMapAI using our ME/CFS datasets. The trained models are available on GitHub, nicknamed DeepMECFS, for the benefit of the ME/CFS research community.
1) Dataset Pre-Processing Module: Handling Sample Imbalance. To ensure uniform learning for each output y, it is crucial to address sample imbalance before fitting the framework. We recommend using customized sample imbalance handling methods, such as Synthetic Minority Over-sampling Technique (SMOTE)110, Adaptive Synthetic (ADASYN)111, or Random Under-Sampling (RUS)112. In our ME/CFS dataset, there is a significant imbalance, with the patient data being twice the size of the control data. To effectively manage this class imbalance, we employed RUS as a random sampling method for the majority class. Specifically, we randomly sampled the majority class 100 times. For each iteration i, a different random subset S_i^{majority} was used. This subset S_i^{majority} of the majority class was combined with the entire minority class S^{minority}. For each iteration i:
S_i^{majority}\subseteq S^{majortiy},\ \ S^{minority}=S^{minority}
S_i=S_i^{majority}\cup S^{minority}
where the combined dataset S_i was used for training at each iteration. This approach allows the model to generalize better and avoid biases towards the majority class, improving overall performance and robustness.
2) Cross-Validation and Model Training. DeepMECFS is the name of the trained BioMapAI model with ME/CFS datasets. We trained on five preprocessed ‘omics datasets, including species abundances (Feature N=118, Sample N=474) and KEGG gene abundances (Feature N=3959, Sample N=474) from the microbiome, plasma metabolome (Feature N=730, Sample N=407), immune profiling (Feature N=311, Sample N=481), and blood measurements (Feature N=48, Sample N=495). Additionally, an integrated ‘omics profile was created by merging the most predictive features from each ‘omics model related to each clinical score (SHAP Methods), forming a comprehensive matrix of 154 features, comprising 50 immune features, 32 species, 30 KEGG genes, and 42 plasma metabolites.
To evaluate the performance of BioMapAI, we employed a robust 5-fold cross-validation. Training was conducted over 500 epochs with a batch size of 64 and a learning rate of 0.0005, optimized through grid search. The Adam optimizer was used to adjust the weights during training, chosen for its ability to handle sparse gradients on noisy data. The initial learning rate was set to 0.01, with beta1 set to 0.9, beta2 set to 0.999, and epsilon set to 1e-7 to ensure numerical stability. Dropout layers with a 50% dropout rate were used after each hidden layer to prevent overfitting, and L2 regularization (\lambda=0.008) was applied to the kernel weights, defined as:
L_{reg}=λ2i=1Nwi2
3) Model Evaluation. To evaluate the performance of the models, we employed several metrics tailored to both regression and classification tasks. The Mean Squared Error (MSE) was used to evaluate the performance of the reconstruction of each object. For each y_i, MSE was calculated as: 
MSE_i=\frac{1}{N}\sum_{j=1}^{N}{\left(y_i^j-{\hat{y}}_i^j\right)^2,\ i=1,2,\ldots,n}
where y_i^j is the actual values, {\hat{y}}_i^j is the predicted values, and N is the number of samples, n is the number of objects. For binary classification tasks (ME/CFS vs control), we utilized multiple metrics including accuracy, precision, recall, and F1 score to enable a comprehensive evaluation of the model's performance.
To evaluate the performance of BioMapAI, we compared its binary classification performance with three traditional machine learning models and one deep neural network (DNN) model. The traditional machine learning models included: 1) Logistic Regression (LR) (C=0.5, saga solver with Elastic Net regularization); 2) Support Vector Machine (SVM) with an RBF kernel (C=2); and 3) Gradient Boosting Decision Trees (GBDT) (learning rate = 0.05, maximum depth = 5, estimators = 1000). DNN model employed the same hyperparameters as BioMapAI, except it did not include the parallel sub-layer, Z_3, thus it only performed binary classification instead of multi-output predictions. The comparison between BioMapAI and DNN aims to assess the specific contribution of the spread-out layer, designed for discerning object-specific patterns, in binary prediction. Evaluation metrics are detailed in Supplemental Table 3.
4) External Validation with Independent Dataset. To validate BioMapAI's robustness in binary classification, we utilized 4 external cohorts28,29,30,31 comprising more than 100 samples. For these external cohorts, only binary classification is available. A detailed summary of data collection for these cohorts is provided in Supplemental Table 4. For each external cohort, we processed the raw data (if available) using our in-house pipeline. The features in the external datasets were aligned to match those used in BioMapAI by reindexing the datasets. The overlap between the features in the external dataset and BioMapAI's feature set was calculated to determine feature coverage. Any missing features were imputed with zeros to maintain consistency across datasets. The input data was then standardized as BioMapAI. We loaded the pre-trained BioMapAI, GBDT, and DNN for comparison. LR and SVM were excluded because they did not perform well during the in-cohort training process. The performance of the models was evaluated using the same binary classification evaluation metrics. Evaluation metrics detailed in Supplemental Table 4.


## 3. BioMapAI Decode Module: SHAP. 
BioMapAI is designed to be explainable, ensuring that it not only reconstructs and predicts accurately but also is interpretable, which is particularly crucial in the biological domain. To achieve this, we incorporated SHapley Additive exPlanations (SHAP) into our framework. SHAP offers a consistent measure of feature importance by quantifying the contribution of each input feature to the model's output.113 
We applied SHAP to BioMapAI to interpret the results, following these three steps:
1) Model Reconstruction. BioMapAI's architecture includes two shared hidden layers - Z^1, Z^2- and one parallel sub-layers - Z_i^3- for each object y_i. To decode the feature contributions for each object y_i, we reconstructed sub-models from single comprehensive model:
Model_i=Z^1+Z^2+Z_i^3,i=1,2,\ldots,n
where n is the number of learned objects.
2) SHAP Kernel Explainer. For each reconstructed model, Model_i, we used the SHAP Kernel Explainer to compute the feature contributions. The explainer was initialized with the model's prediction function and the input data X:
explainer_i=shap.KernelExplainer\left(Model_i.predict,X\right),\ i=1,2,\ldots,n
Then SHAP values were computed to determine the contribution of each feature to y_i:
\phi_i=explainer_i\left(X\right),i=1,2,\ldots,n
The kernel explainer is a model-agnostic approach that approximates SHAP by evaluating the model with and without the feature of interest and then assigning weights to these evaluations to ensure fairness. For each model_i, with each feature j:
\phi_i^j\left(f,x\right)=\ \sum_{S_i\subseteq N_i\\{j}}{\frac{\left|S_i\right|!\left(m-\left|S_i\right|-1\right)!}{m!}\left(Model_i\left(S_i\cup j\right)-Model_i\left(S_i\right)\right)}\bigm=\ \frac{1}{m}\sum_{S_i\subseteq N_i\\{j}}{\binom{m-1}{m-\left|S_i\right|-1}^{-1}\left(Model_i\left(S_i\cup j\right)-Model_i\left(S_i\right)\right)},\ i=1,2,\ldots,n
where n is the number of learned objects, m is the total number of features, \phi_i^j is the Shapley value for feature j in model_i, N_i is the full set of features in model_i, S_i is the subset of features not including feature j, Model_i\left(S_i\right) is the model prediction for the subset S_i. The SHAP value matrix, \phi_i, were further reshaped to align with the input data dimensions.
3) Feature Categorization. Analyzing the SHAP value matrices, \left[\phi_1,\ \phi_2,\ \ldots,\phi_n\right], features can be roughly assigned to two categories: shared features - important to all outputs; or specific features - specifically important to individual outputs. We set the cutoff at 75%, where features consistently identified as top contributors in 75% of the models were classified as shared important features, termed disease-specific biomarkers. Features that were top contributors in only a few models were classified as specific important features, termed symptom-specific biomarkers.
By reconstructing individual models, Model_i, for each object, y_i, and applying SHAP explainer individually, we effectively decoded the contributions of input features to BioMapAI's predictions. This method allowed us to categorize features into shared and specific categories—termed as disease-specific and symptom-specific biomarkers—providing novel interpretations of the ‘omics feature contribution to clinical symptoms. 

## 4. Packages and Tools. 
BioMapAI was constructed by Tensorflow(v2.12.0) and Keras(v2.12.0). ML models were from scikit-learn(v 1.1.2).
