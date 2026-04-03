# 🎓 Student GPA Predictor (ML + Gradio)

A Machine Learning web application that predicts a student's GPA based on demographic, academic, and lifestyle factors. This project demonstrates an end-to-end ML workflow from model training to deployment using a simple interactive UI.
# 📸 Demo
# Link: https://huggingface.co/spaces/arfat16/Student-result-predictor
<img width="1707" height="903" alt="image" src="https://github.com/user-attachments/assets/1f95b09d-bfa9-43a8-a4fc-7cd50394af3e" />
<img width="1722" height="832" alt="image" src="https://github.com/user-attachments/assets/e75bfe2d-604d-4aca-a625-9d4e942e02c8" />


# 🚀 Features
 1. Predict student GPA using a trained ML model
 2. Built with a Random Forest Pipeline
 3. Interactive UI using Gradio
 4. Handles categorical + numerical features via pipeline

# Input Features
1. Gender
2. Age
3. Address (Urban/Rural)
4. Family Size
5. Parental Status
6. Mother’s Education
7. Father’s Education
8. Mother’s Job
9. Father’s Job
10. Relationship Status
11. Smoker (Yes/No)
12. Tuition Fee
13. Time with Friends
14. SSC Result

 # Tech Stack
* Python
* Scikit-learn
* Pandas / NumPy
* Gradio

#  Installation & Setup <br/>
1️⃣ Clone the repository <br/>
git remote add origin https://github.com/yunusarfat/student-result-predictor-ml.git <br/>
2️⃣ Create virtual environment <br/>
python -m venv venv <br/>
venv\Scripts\activate   # Windows <br/>
3️⃣ Install dependencies <br/>
pip install -r requirements.txt <br/>
▶️ Run the App <br/>
python app.py <br/>
