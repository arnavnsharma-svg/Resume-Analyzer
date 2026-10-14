from sklearn.model_selection import train_test_split
import re 
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
resumes = [
    "Python machine learning pandas SQL",
    "HTML CSS JavaScript React frontend",
    "Java Spring Boot API development",
    "Python statistics data visualization",
    "React HTML CSS frontend developer",
    "Java backend Spring database API",
    "Machine learning Python model training",
    "JavaScript React web application",
    "Java Spring Boot backend developer",
    "Python SQL data analysis",
    "HTML CSS frontend web design",
    "Java API database backend development"
]
categories = [
    "Data Science", "Web Development", "Java Developer",
    "Data Science", "Web Development", "Java Developer",
    "Data Science", "Web Development", "Java Developer",
    "Data Science", "Web Development", "Java Developer"
]
def text_cleaner(text):
    text=text.lower()
    text=re.sub('\s+'," ",text)
    text=re.sub(r'[^A-Za-z\s]','',text)
    return text.strip()
resumes=[text_cleaner(resume) for resume in resumes]
X_train,X_test,Y_train,Y_test=train_test_split(resumes,categories,test_size=0.25,random_state=48,stratify=categories)
pipeline=Pipeline([('tfidf',TfidfVectorizer()),('model',RandomForestClassifier(random_state=48,n_estimators=100))])
pipeline.fit(X_train,Y_train)
predict=pipeline.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, predict))

print("\nClassification Report:")
print(classification_report(
    Y_test, predict, zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    Y_test, predict,
    labels=pipeline.classes_
))
print("\nCategory order:", pipeline.classes_)
print('F1 score',f1_score(Y_test, predict,average='macro'))
print("Precision:", precision_score(Y_test, predict, average="macro", zero_division=0))
print("Recall:", recall_score(Y_test, predict, average="macro", zero_division=0))
joblib.dump(pipeline,"resume_analyzer.joblib")
print('model saved successfully')
loaded_model=joblib.load('resume_analyzer.joblib')
new_resume = "Python developer with SQL and machine learning skills"
prediction=loaded_model.predict([text_cleaner(new_resume)])
print('Predicted resume',prediction[0])