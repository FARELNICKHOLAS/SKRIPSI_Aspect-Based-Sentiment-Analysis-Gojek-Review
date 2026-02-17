```markdown
# ABSA Gojek Review Dashboard (RoBERTa)

This project is a Streamlit-based web application for Aspect-Based Sentiment Analysis (ABSA) on Gojek app reviews using fine-tuned RoBERTa models.  
The system performs:
- Text preprocessing (cleaning + slang normalization)
- Aspect classification
- Sentiment classification (Positive / Negative)
- Single and Batch (CSV) prediction

---

## Project Structure

Make sure your folder structure looks like this:

```

skripsizip/
│
├── app.py
├── requirements.txt
├── slang_dict.json
│
├── roberta_aspect/
│   ├── roberta_aspect_model.h5
│   └── roberta_tokenizer_aspect.zip
│
├── roberta_sentiment/
│   ├── roberta_sentiment_model.h5
│   └── roberta_tokenizer_sentiment.zip
│
├── dataset/
└── notebook/

````

> The model files are NOT included in the repository and must be downloaded manually from Google Drive (link will be provided by the author).

---

## 1. Create Virtual Environment (Recommended)

It is strongly recommended to use a virtual environment to avoid dependency conflicts.

### Windows (PowerShell / CMD)

```bash
cd D:\skripsizip
python -m venv venv
venv\Scripts\activate
````

### macOS / Linux

```bash
cd /path/to/skripsizip
python3 -m venv venv
source venv/bin/activate
```

If the environment is active, you will see `(venv)` in your terminal.

---

## 2. Install Dependencies

Install all required libraries using:

```bash
pip install -r requirements.txt
```

### requirements.txt

```
streamlit
pandas
tensorflow
transformers
tf-keras
```

---

## 3. Download Pretrained Models (IMPORTANT)

This project requires two pretrained models:

* RoBERTa Aspect Model
* RoBERTa Sentiment Model

You must download them from the Google Drive link that will be provided.

Example (replace with your actual link):

```
Google Drive: [https://drive.google.com/drive/folders/1x5tlLFwSEZYfFNl_xZ7v2I_LNJrkieuG?usp=sharing]
```

---

## 4. Where to Place the Downloaded Models

After downloading, extract and place the files EXACTLY as follows:

### A. Aspect Model

Put these files inside:

```
skripsizip/roberta_aspect/
```

Required files:

* `roberta_aspect_model.h5`
* `roberta_tokenizer_aspect.zip`

Final structure:

```
roberta_aspect/
├── roberta_aspect_model.h5
└── roberta_tokenizer_aspect.zip
```

### B. Sentiment Model

Put these files inside:

```
skripsizip/roberta_sentiment/
```

Required files:

* `roberta_sentiment_model.h5`
* `roberta_tokenizer_sentiment.zip`

Final structure:

```
roberta_sentiment/
├── roberta_sentiment_model.h5
└── roberta_tokenizer_sentiment.zip
```

> Do NOT rename the files. The application loads them using fixed paths.

---

## 5. Slang Dictionary

The system uses a slang normalization dictionary:

```
slang_dict.json
```

This file must remain in the project root directory because it is loaded with:

```python
load_slang_dict("slang_dict.json")
```

---

## 6. Run the Application

After everything is set up (venv, dependencies, and models), run:

```bash
streamlit run app.py
```

Or if your main file is named differently (e.g., `app.py`):

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```
http://localhost:8501
```

---

## App Features:

### 1. Single Review Prediction

* Input one Gojek review manually
* Outputs:

  * Preprocessed text (optional debug)
  * Predicted Aspect
  * Predicted Sentiment
  * Confidence Score

### 2. Batch CSV Prediction

Upload a CSV file with the required column:

```
review
```

Example:

```csv
review
Aplikasinya bagus tapi sering error
Driver ramah dan cepat
```

The system will automatically:

* Process all reviews
* Add aspect, sentiment, and confidence columns
* Allow downloading the result as CSV

---

## 🔧 Preprocessing Pipeline

The text preprocessing includes:

1. Case folding (lowercasing)
2. Special character & punctuation removal
3. Whitespace normalization
4. Repeated character normalization (e.g., "bagussss" → "baguss")
5. Slang normalization using `slang_dict.json`

---

## Notes:

* Ensure TensorFlow is compatible with your Python version (recommended: Python 3.9–3.11)
* First model loading may take longer due to tokenizer extraction
* Do not change folder names (`roberta_aspect` and `roberta_sentiment`) because they are hardcoded in the application
* The tokenizer ZIP files will be automatically extracted during runtime

---

## Author

**Wayan Farel Nickholas Sadewa**
Thesis Project – Aspect-Based Sentiment Analysis on Gojek Reviews using RoBERTa

```
```
