# BreastCancerDiagnosis_DeepLearning
Deep Learning Neural Networks: identifying benign vs malignant cancers and distinguishing between tumor types by applying Binary classification and Multi-class classification algorithms, respectively.

## Breast Cancer Diagnosis Using Deep Learning
**Introduction:** <>
This project leverages deep learning to assist in the diagnosis of breast cancer through image classification. Using the BreaKHis dataset, which contains high-resolution microscopic images of breast tissue, the objective is to build and evaluate models that can accurately classify breast cancer images into two categories: benign and malignant (binary classification), and further classify them into specific tumor types (multi-class classification). <br>
Breast cancer is a significant global health concern, and accurate diagnosis is crucial for timely and effective treatment. This project aims to explore advanced deep learning techniques to improve diagnostic precision and support medical professionals in distinguishing between benign and malignant tissues.

**Data Description:**
- BreaKHis dataset: consists of high-resolution microscopic images of breast tissue, where each image is accompanied by metadata providing essential information for model training and evaluation. 

Key features:

- **Path_to_image:** the file path to the histopathological image

- **Benign or Malignant:** binary label indicating the tumor type

- **Cancer Type:** specific tumor classification (8 possible types)

- **Magnification:** the magnification level at which the image was captured

**Objectives:**

- **Stage 1:** develop a binary classification model to differentiate between benign and malignant breast tissue images

- **Stage 2:** extend the model to multi-class classification, predicting the specific tumor type from the following categories:
  - Adenosis
  - Tubular Adenoma
  - Phyllodes Tumor
  - Ductal Carcinoma
  - Lobular Carcinoma
  - Mucinous Carcinoma
  - Papillary Carcinoma
  - Fibroadenoma

**Methodology:** <br>
The project adopts a data-driven approach combining image pre-processing, deep learning model development and performance evaluation:

**1. Data Pre-processing:**
- Image normalization and resizing
- Data augmentation (e.g., rotation, flipping, and contrast adjustment)
- Handling class imbalance through oversampling and other techniques

**2. Model Development:**
- Implement and fine-tune **Convolutional Neural Networks (CNNs)** for image classification
- Explore **pre-trained models** (e.g., VGG16, ResNet, EfficientNet) using transfer learning
- Optimize hyperparameters to enhance model performance

**3. Model Evaluation:**
- Use metrics such as **Accuracy**, **Precision**, **Recall**, **F1-score**, and **AUC-ROC**
- Perform error analysis and interpret the most common misclassifications using confusion matrices
- Compare multiple model architectures and select the best-performing one

**4. Experimental Setup:**
- Train models on the **BreaKHis** dataset using **Python 3**
- Evaluate both intermediate and final models
- Document the decision-making process and justify the best approach

## Technologies Used:
Programming Language: **Python 3**

Libraries/Frameworks:
- **TensorFlow & Keras:** Model development and training
- **Pandas & NumPy:** Data manipulation and preprocessing
- **Matplotlib & Seaborn:** Data visualization and analysis
- **OpenCV:** Image processing and augmentation
- **Scikit-learn:** Model evaluation and performance metrics
- **Google Colab/Jupyter Notebook:** Interactive model development

## The repository is organized as follows:
- Notebooks (.ipynb):
  - Binary_Classifier
  - Multiclass_Classifier
  - Stage_1_Undersampling
  - Stage_2_Undersampling
  - Stage1_Model_Undersampling
- Project Statement
- Report
