# Resume Analyzer using Machine Learning

A machine learning project that classifies resume text into one of three professional categories: **Data Science, Web Development, and Java Developer**. It uses TF-IDF for text feature extraction and a Random Forest classifier for prediction.

## Overview

The Resume Analyzer demonstrates a basic NLP and machine learning workflow, from text preprocessing and feature extraction to model training, evaluation, saving, and loading for inference.

The model predicts a category based on the skills and keywords present in the supplied resume text.

## Features

- **Text preprocessing:** Converts text to lowercase, removes extra whitespace and non-alphabetic characters.
- **TF-IDF vectorization:** Converts resume text into numerical features.
- **Machine learning classification:** Uses a Random Forest classifier to categorize resumes.
- **Model evaluation:** Calculates accuracy, precision, recall, F1-score, classification report, and confusion matrix.
- **Model persistence:** Saves the trained pipeline using Joblib and reloads it for predictions.
- **Resume classification:** Predicts a category for new resume text.

## Technologies Used

- Python
- Scikit-learn
- TF-IDF Vectorizer
- Random Forest Classifier
- Pandas-free text processing with Python's `re` module
- Joblib

## Project Workflow

1. Prepare a small labeled dataset containing resume-related text and categories.
2. Clean the text by converting it to lowercase, normalizing whitespace, and removing non-alphabetic characters.
3. Split the dataset into training and testing sets using a stratified split.
4. Extract text features using `TfidfVectorizer`.
5. Train a `RandomForestClassifier` within a Scikit-learn pipeline.
6. Evaluate predictions using multiple classification metrics.
7. Save the trained pipeline to `resume_analyzer.joblib`.
8. Load the saved model and classify new resume text.

## Categories

| Category | Example keywords |
|---|---|
| Data Science | Python, machine learning, pandas, SQL, statistics |
| Web Development | HTML, CSS, JavaScript, React |
| Java Developer | Java, Spring Boot, APIs, databases |

These examples represent the categories in the project's current sample dataset.

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/arnavnsharma-svg/REPOSITORY-NAME.git
cd REPOSITORY-NAME
```

Replace `REPOSITORY-NAME` with your actual GitHub repository name.

**2. Install the dependencies**

```bash
pip install scikit-learn joblib
```

**3. Run the Python script**

```bash
python resume_analyzer.py
```

Replace `resume_analyzer.py` with your actual Python filename if it differs.

## Example

```python
new_resume = "Python developer with SQL and machine learning skills"

prediction = loaded_model.predict([
    text_cleaner(new_resume)
])

print("Predicted resume:", prediction[0])
```

The model predicts one of the three supported categories based on the learned text patterns.

## Model Evaluation

The project evaluates the classifier using:

- Accuracy
- Macro-averaged precision
- Macro-averaged recall
- Macro-averaged F1-score
- Classification report
- Confusion matrix

The actual metric values are printed when the script runs. They are not specified here because they depend on the trained model's predictions.

## Current Limitations

- The model uses a small sample dataset of 12 labeled text examples.
- The categories are limited to Data Science, Web Development, and Java Developer.
- The current input consists of plain text; PDF/DOCX resume parsing and file uploads are not implemented in the provided code.
- The model identifies text patterns rather than comprehensively evaluating a candidate's qualifications.
- Performance on real-world resumes may differ significantly from results on the sample dataset.

## Future Improvements

- Train on a larger and more diverse resume dataset.
- Add PDF and DOCX text extraction.
- Support additional job categories.
- Compare Random Forest with other classification algorithms.
- Tune hyperparameters and evaluate performance on a larger test set.
- Build a web interface for submitting resume text and displaying predictions.

## Author

**Arnav Sharma**

- GitHub: [arnavnsharma-svg](https://github.com/arnavnsharma-svg)
- LinkedIn: [Arnav Sharma](https://www.linkedin.com/in/arnav-sharma-5b2144193/)

## Disclaimer

This is an educational machine learning project intended to demonstrate text classification. Its predictions should not be treated as a definitive assessment of a candidate's suitability for a job.
