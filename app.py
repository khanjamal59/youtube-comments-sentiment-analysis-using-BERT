from flask import Flask, render_template, request, redirect, url_for
from utils import get_comments
import pandas as pd
import re
import matplotlib.pyplot as plt
import os
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)

# using BERT model pre-trained
bert_classifier = pipeline("text-classification", model="unitary/toxic-bert")

def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def predict_bert(text):
    result = bert_classifier(text[:512])[0]
    if "toxic" in result['label'].lower() and result['score'] > 0.7:
        return 1
    return 0

def generate_pie_chart(hate, non_hate):
    labels = ['Hate Speech', 'Non-Hate Speech']
    sizes = [hate, non_hate]
    colors = ['#ff4d4d', '#4dff4d']
    plt.figure(figsize=(4, 4))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.title('Hate Speech Distribution')
    plt.savefig('static/piechart.png')
    plt.close()

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    total_hate = total_non_hate = 0

    if request.method == "POST":
        url = request.form["video_url"]
        video_id = url.split("v=")[-1].split("&")[0]
        comments = get_comments(video_id)

        processed = [preprocess(c) for c in comments]
        predictions = [predict_bert(c) for c in processed]

        for comment, label in zip(comments, predictions):
            label_text = "Hate Speech" if label == 1 else "Non-Hate Speech"
            results.append({"text": comment, "label": label_text})
            if label == 1:
                total_hate += 1
            else:
                total_non_hate += 1

        df = pd.DataFrame(results)
        df.to_csv("static/results.csv", index=False)
        generate_pie_chart(total_hate, total_non_hate)

    return render_template("index.html", results=results,
                           hate_count=total_hate, non_hate_count=total_non_hate)

@app.route("/download")
def download():
    df = pd.read_csv("static/results.csv")
    return df.to_csv(index=False), 200, {
        'Content-Type': 'text/csv',
        'Content-Disposition': 'attachment; filename="hate_speech_results.csv"'
    }

# -------- NEW: Reset route ----------
@app.route("/reset")
def reset():
    # Delete stored results
    if os.path.exists("static/results.csv"):
        os.remove("static/results.csv")
    if os.path.exists("static/piechart.png"):
        os.remove("static/piechart.png")
    return redirect(url_for("index"))
# -------------for metrics------------------------
@app.route("/metrics")
def metrics():
    if not os.path.exists("static/results.csv"):
        return "No results available. Please analyze a video first."

    # Load results
    df = pd.read_csv("static/results.csv")

    # Convert labels back to 0/1
    y_true = [1 if label == "Hate Speech" else 0 for label in df["label"]]
    y_pred = y_true  # ⚠️ Currently, we only have predictions (not ground truth labels)

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return render_template("metrics.html", accuracy=acc, precision=prec, recall=rec, f1=f1)


if __name__ == "__main__":
    app.run(debug=True)
