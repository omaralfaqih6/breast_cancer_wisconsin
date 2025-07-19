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
## Clinical doctors
Their interest in the project is knowing the exact features required to predict the cancer will increase the accuracy of diagnosis whether its malignant or benign. Which will lead to better treatment/management for the patients. 
b. Hospital Management - It will reduce the cost in diagnosing the patient. 
c. Hospital's Legal department - Avoiding the lawsuits against doctors for consequences of false diagnosis.
d. Medical Equipment Manufactors - Enabling manufacturers in building more optimized diagnostic panels for collecting the data.
e. Patients - Interested in being accurately diagnosed to avoid the unnecessary health and psychological effects of false diagnostics.
f. Ministry of Health - The accurate disgnostics will reduce the cost and the expenses paid by Public health budget.


# Project Goals
The goal of the project is to identify features that are most predictive of malignant cancer in the Wisconsin Breast Cancer dataset. The dataset features approximately 30 predictor variables associated with each sample. Therefore, The project team shall aim to: 
 - Explore various models to identify 3 models best fit for the problem.
 - Build 3 models using different machine learning techniques and compare their performance.
 - Identify 3-5 features that are most definitive contributor to the model performance from multiple different models.

# Project Report
- Refine the project and identify which features for their significance(top 3 or 5), and the reason & the criteria on choosing.
- We read about 5 models listed below to choose 3 models only for our project. These models are the best fit for small datasets:
  - logistic regression
  - Support Vector Machine (SVM)
  - Random Forest Model (RFM)
  - AutoGloun
  - LightGBM
- Among these 5 models considered, the project team chose 3 models listed below:
 - logistic regression
 - Support Vector Machine (SVM)
 - Random Forest Model (RFM)

> [!NOTE]
> The project team read this article that mentioned the comparison between the models that are most suitable and fit for this dataset. The article is found [here](https://www.data-cowboys.com/blog/which-machine-learning-classifiers-are-best-for-small-datasets)

- List the names of the 3 models - logistic regression, Support Vector Machine (SVM),  & Random Forest Model (RFM) chosen.
- List the accuracy of each model - @SB
- The choice of a model (How and why)
- Work on finding the top-5 features contributed to the predictions
- Include images and graphes/visuals

Project's Tools and 

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