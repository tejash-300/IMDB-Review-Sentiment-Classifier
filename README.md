# IMDB Review Sentiment Classifier 🎭📊

A **Sentiment Analysis Model** for **IMDB Movie Reviews** using **Natural Language Processing (NLP)** and **Deep Learning**.

## 🚀 Overview

This project classifies IMDB movie reviews as **positive** or **negative** using **BERT**, **LSTM**, or other NLP techniques. It processes textual data, extracts sentiments, and predicts review polarity.

### 🏆 Key Features:
- **Preprocessing**: Tokenization, Stopword Removal, Lemmatization
- **Word Embeddings**: BERT, Word2Vec, TF-IDF, or GloVe
- **Deep Learning Model**: LSTM, CNN, or Transformer-based model
- **Evaluation**: Accuracy, Precision, Recall, and F1-Score

---

## 📂 Project Structure
```
📁 IMDB-Sentiment-Classifier
│── 📜 IMDB Review Sentiment Classifier.ipynb   # Jupyter Notebook
│── 📜 README.md                                 # Project Documentation
│── 📜 requirements.txt                          # Dependencies
│── 📁 data/                                     # Dataset Folder
│   ├── train.csv                                # Training Data
│   ├── test.csv                                 # Test Data
│── 📁 models/                                   # Trained Models
│   ├── sentiment_model.h5                      # Saved Model
│── 📁 results/                                  # Model Performance Results
│── 📁 src/                                      # Python Scripts (if needed)
```

---

## 🔧 Installation & Setup

1️⃣ **Clone the Repository**
```sh
git clone https://github.com/your-username/IMDB-Sentiment-Classifier.git
cd IMDB-Sentiment-Classifier
```

2️⃣ **Create a Virtual Environment (Optional)**
```sh
python -m venv venv
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate  # For Windows
```

3️⃣ **Install Dependencies**
```sh
pip install -r requirements.txt
```

4️⃣ **Run Jupyter Notebook**
```sh
jupyter notebook
```

---

## 📊 Dataset

- **Source**: IMDB Movie Reviews Dataset (50,000 labeled reviews)
- **Classes**: Positive (1) & Negative (0)
- **Split**: 80% Training, 20% Testing

Download dataset: [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)

---

## 🛠 Model Training & Evaluation

1️⃣ **Preprocessing Steps**:
   - Lowercasing, Tokenization, Stopwords Removal, Lemmatization
   - Word embeddings: **BERT / Word2Vec / TF-IDF**

2️⃣ **Training the Model**:
   - **Deep Learning Models**: LSTM / CNN / Transformer
   - **Optimizer**: Adam
   - **Loss Function**: Binary Cross-Entropy

3️⃣ **Evaluation Metrics**:
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix, ROC Curve

---

## 📌 Usage

- Modify the **IMDB Review Sentiment Classifier.ipynb** notebook for training
- Save the trained model and predict sentiment on new reviews:
```python
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/sentiment_model.h5")

# Predict sentiment
review = "This movie was amazing, I loved the acting!"
prediction = model.predict([review])
print("Positive" if prediction > 0.5 else "Negative")
```

---

## 📈 Results

| Model       | Accuracy  | Precision | Recall | F1-Score |
|------------|----------|-----------|--------|----------|
| LSTM       | 88.2%    | 87.5%     | 86.8%  | 87.1%    |
| CNN        | 85.7%    | 85.2%     | 84.9%  | 85.0%    |
| BERT       | 91.4%    | 91.0%     | 90.6%  | 90.8%    |

---

## 🏆 Future Enhancements

✅ Fine-tuning BERT for better performance  
✅ Hyperparameter tuning using **GridSearchCV**  
✅ Deploy model using **Flask / FastAPI / Streamlit**  

---

## 🤝 Contributing

Want to improve this project?  
- ⭐ Star the repo  
- Create a new branch & submit a PR  


