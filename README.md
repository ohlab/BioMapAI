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
   pip install numpy pandas tensorflow
   ```

2. **Clone or Download This Repository**

   This repository should include:
   - `BioMapAI.py`: Contains the classes and methods (OmicScoreModel, ScoreLayer, etc.).
   - `example_data/`: Folder containing `train_data.csv` and `test_data.csv`.
   - `BioMapAI_Training_Tutorial.ipynb`: Detailed notebook tutorial.

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
   - Modify `BioMapAI.py` to create custom network architectures or loss functions.
   - Integrate advanced data preprocessing or feature engineering techniques.

## Further Details

For an in-depth guide, check out the [BioMapAI_Training_Tutorial.ipynb](BioMapAI_Training_Tutorial.ipynb). It covers:

- Data loading and organization  
- Model instantiation and training procedures  
- How to evaluate intermediate and final predictions  
- Strategies for adjusting the model to improve performance  

---


## To Load and use our pretrained model for ME/CFS: DeepMECFS
