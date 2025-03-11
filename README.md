# BioMapAI
The primary goal of BioMapAI is to connect high-dimensional biology data, $X$ to mixed-type output matrix, $Y$. Unlike traditional ML or DL classifiers that typically predict a single outcome, $y$, BioMapAI is designed to learn multiple objects, $Y=\left[y_1,\ y_2,\ \ldots,y_n\right]$, simultaneously within a single model. This approach allows for the simultaneous prediction of diverse clinical outcomes - including binary, categorical, continuous variables - with ‘omics profiles, thus address disease heterogeneity by tailoring each patient’s specific symptomatology.

## Recommended System Requirements

To run BioMapAI and DeepMECFS efficiently, we recommend the following hardware and software setup.

### Hardware Requirements
- **CPU**: Minimum Intel Core i7 / AMD Ryzen 7, Recommended Intel Xeon / AMD Threadripper
- **RAM**: Minimum 16GB, Recommended 32GB+ for large datasets
- **GPU**: No GPU required
- **Storage**: SSD recommended for faster I/O

### Software Requirements
**Version**:
- Python Version: 3.9.13
- TensorFlow Version: 2.12.0
- Pandas Version: 1.5.0
- Numpy Version: 1.24.0
**OS**: Linux/macOS/Windows (Linux recommended for best performance)

### Runtime Estimates
- **Training BioMapAI Model**: Total execution time: 23.61 seconds on CPU: x86_64 with Total RAM: 810.20 GB
- **Inference with Pretrained DeepMECFS Model**: Total execution time: 10.36 seconds econds on CPU: x86_64 with Total RAM: 810.20 GB

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


We have used BioMapAI to build **pretrained** models specifically for ME/CFS omics data, called **DeepMECFS**.We trained BioMapAI on gut microbiome data (species abundance and KEGG gene abundance), plasma metabolome, high-throughput immune flow cytometry data, Quest lab measurements, and a combined omics file containing key features from all datasets. These models are located in the folder `pretrained_model_DeepMECFS/` and can be applied directly to new ME/CFS datasets. Here we use one of public metabolome datasets as an example to walk through how to load and use our pretrained models.

### DeepMECFS Contents

- `DeepMECFS_metabolome/`: Directory containing the trained TensorFlow model.
- `Y2y_metabolome/`: Secondary model for converting intermediate features (`Y`) into final ME/CFS classification.
- `metabolome_feature_metadata.csv`: Required features and metadata for alignment with your dataset.

### How to Use the Pretrained Model

1. **Install/Clone** the repository containing the `pretrained_model_DeepMECFS/` folder.
2. **Prepare Your Data**:
   - Ensure your metabolomics data columns match the names (or COMP_IDs) in `metabolome_feature_metadata.csv`.
   - Scale or normalize your data consistently (e.g., via `StandardScaler`).
3. **Run the Tutorial**:
   - Open `DeepMECFS_Tutorial.ipynb` (or equivalent notebook/script).
   - Follow each step to:
     1. Load the pretrained models (`DeepMECFS_metabolome/` and `Y2y_metabolome/`).
     2. Align your dataset columns to the model’s expected features.
     3. Generate predictions (ME/CFS vs. Control).
     4. Evaluate performance metrics (accuracy, AUC, precision, etc.).
4. **Interpret Results**:
   - The model outputs a probability (`0 to 1`) for ME/CFS classification.
   - You can threshold this probability (e.g., 0.5) to get a binary label (`CFS` vs. `Control`).
5. **Explore Further**:
   - You can experiment with different preprocessing or consider re-training parts of the pipeline if your data differs significantly from the original study.

### Reference

The metabolomics data used to train DeepMECFS is described in:
> Arnaud Germain, et al. “Plasma metabolomics reveals disrupted response and recovery following maximal exercise in myalgic encephalomyelitis/chronic fatigue syndrome.” **JCI Insight**. 2022;7(9):e157621.  
> [DOI: 10.1172/jci.insight.157621](https://doi.org/10.1172/jci.insight.157621)

For detailed instructions, see the [Pretrained_DeepMECFS_Tutorial.ipynb](Pretrained_DeepMECFS_Tutorial.ipynb). It includes code snippets for loading the data, aligning it to the model’s features, and running inference.


### License

This project is provided under the [MIT License](./LICENSE) . Feel free to modify or redistribute under its terms.

---
