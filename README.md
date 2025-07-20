# Improving Breast Cancer Diagnostics Project

Our team is dedicated throughout this project to improve the breast cancer diagnostics leveraging machine learning capabilities provided by the latest technologies. 
Our project will have the below scope:
- Choose 3 machine learning models and evaluate thier capabilities in predicting the diagnostics of patients as benign or malignant.
- Investigate and engineer a set of features best to perform accuracy and optimized prediction.
- Provide a list of recommendations and action items to project stakehlders for them leveraging the data-driven solution to enhance cancer diagnostics.

> [!NOTE]
> This initiative is considered to be a research project. It is a prelude to a pilot project to be implemented at hospitals and cancer clinics with the consideration of partial or full implementation of recommendations.

## Content

Business Case
Stakeholders
Project Goals & Milestones
Project Reporting
Considerations and Risk Management
Recommendations
Team Members

# Business Case
Our business case is focused on leveraging data using machine learning models and methodologies to investigate and find the features of a breast mass which are the most predictive of breast cancer diagnosis.

## Breast Cancer Wisconsin - Dataset Analysis
This dataset is representing 30 features for 569 samples derived from digitized images of fine-needle aspirate (FNA) tests performed on breast cancer. These features are numerical and continiuos, also, our team had confirmed that none of these features have any missing values.
Our target variable (the depednent variable) is Diagnosis variable which has either of the below values:
- M: for malignant
- B: for benign

> [!NOTE]
> The dataset was downloaded and its analysis were done based on information provided in [Breat Cancer Wisconsin (Diagnostic) paper](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

# Stakeholders
After investigating and analyzing the medical field, we had found that below are the primary stakeholders benefitting from this initiative:
|Stakeholder|Interest|
|:-----:|------|
|Patients|Patients are at the top of the list as their life is depedent on the accuracy of the diagnosis. The more accuraly diagnozed, patients will receive the required treatment in the shortest period of time, and may be the reason for saving their lives in certain cases.|
|Clinical doctors|Their interest in the project is knowing the most important features required to predict the cancer will increase the accuracy of diagnosis whether its malignant or benign. Which will lead to better treatment/management for the patients.|
|Medical Equipment Manufactors|This initiative will enable the manufacturers building more optimized diagnostic panels for collecting the data.|
|Hospital Management|The management are interested in reducing the diagnostics costs and consequentally the patient's treament. This goal of this initiative is firmly aligned with this interest as finding the most important features will result in a more optimized diagnostic approach. This will consequently lead to more accurate diagnostics, better treatment and reduced cost.|
|Hospital's Legal department | The more optimized diagnostics approach will increase the accuracy of the diagnostic, thereby, avoiding the lawsuits against doctors and the hospital due to consequences of false diagnosis.|
|Ministry of Health|The accurate disgnostics will reduce the cost and the expenses paid by Public health budget.|

# Project Goals
The goal of the project is to identify features that are most predictive of malignant cancer in the Wisconsin Breast Cancer dataset. The dataset features approximately 30 predictor variables associated with each sample. Therefore, The project team shall aim to: 
 - Explore various models to identify 3 models best fit for the problem.
 - Build 3 models using different machine learning techniques and compare their performance.
 - Identify 3-5 features that are most definitive contributor to the model performance from multiple different models.

# Project Report (Finding & Results)
In this section, our team had listed the project milestones along with the results and the methodologies that we had used to follow though and meet the project goals.

## Project Flow Chart
To have a better visualization of the project phases and milestones, our project team had created the below flow chart:

```mermaid
graph TD;
    A(["Models Selection"]);
    B(["Data Analysis"]);
    C(["Models Development"]);
    D(["Logistic Regression"]);
    E(["Random Forest Model (RFM)"]);
    F(["Support Vector Machine (SVM)"]);
    A-->B;
    B-->C;
    C-->D;
    C-->E;
    C-->F;
    D-->G{"Choose Best ML Model"};
    E-->G;
    F-->G;
    G-->H(["Top 5 Important Features"]);
    
```


## Models Selection
Our project team had complete a research on the qualified machine learning models to be using for classifying the dataset. The team had listed 5 potential ML models listed below:
  - logistic regression
  - Support Vector Machine (SVM)
  - Random Forest Model (RFM)
  - AutoGloun
  - LightGBM

Among these listed models, the team had chosen 3 models for our project. These models are the best fit for small datasets and we will conduct our research and analysis in the upcoming sections:
 - logistic regression
 - Support Vector Machine (SVM)
 - Random Forest Model (RFM)

> [!NOTE]
> The project team read this article that mentioned the comparison between the 5 models that are most suitable and fit for this dataset. The article is found [here](https://www.data-cowboys.com/blog/which-machine-learning-classifiers-are-best-for-small-datasets)

## Data Analysis
The team had conducted analysis on the dataset which includes the below:
1. Ensure that the dataset is clean and had no missing values.
2. Demonstrated the relationships between the features themselves and also with the target variable.

## ML Models Development
The project team had development and trained the three models on the dataset. Every section below will show the Python code used for every model and the performance criteria used to select the best performing model.

### Logistic Regression

### Support Vector Machine (SVM)

### Random Forest Model (RFM)

### Winning ML Model
This section defined the criteria the project team had used to select the model (The How!), and provide the technical justification (The Why!).

## The Most Important Features - Top 5


# Risks Identified & Considerations
Dataset findings:
- Working with sample dataset
- Any features missing thaqt can be useful
- The method of acquisition the data, ease of measuring the data. 
- 
Missing Human behavior and deitary issues

Project risks including uncertainty, choosing the model, the evaluation methods, the more different model we evaluate, the better the risk management we'll follow. 
The depednency of the data on the models.

# Recommendations
- Provide a list of actionable items to be addressed by your audience, for example, for doctors.
- Based on the importance of features, highlight the features and what should be done or considered in measuring them.

# Team members
Below are our team members:
|Name|GitHub|Roles|Contribution Video|
|:--:|:--:|:--:|:--:|
|Sanjeev Budhathoki|https://github.com/budsans|Models Development|
|Omar Alfaqih|https://github.com/omaralfaqih6/|Result Analysis, Documentation, Model Optimization|
|Azhar Hasan|https://github.com/azharhasan|Model Optimization, Risk Analysis|

# Further Readings
Below are the links mentioned in this article for further readings and advanced research.
