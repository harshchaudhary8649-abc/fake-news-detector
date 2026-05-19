from flask import Flask, render_template, request, redirect, url_for, session, send_file
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4

# mera 
app = Flask(__name__)
app.secret_key = "fake_news_detector_secret_key"

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load model info
if os.path.exists("model_info.json"):
    with open("model_info.json", "r", encoding="utf-8") as file:
        model_info = json.load(file)
else:
    model_info = {
        "best_model": "Unknown",
        "accuracy": 0,
        "dataset_size": 0,
        "real_samples": 0,
        "fake_samples": 0,
        "classification_report": "Not available"
    }

history_file = "history.json"

fake_keywords = [
    "shocking", "guarantee", "100%", "100 percent", "click here",
    "miracle", "viral", "secret", "unbelievable", "earn money fast",
    "instant", "doctors hate", "one trick", "free money", "rumor",
    "spam", "leaked", "hidden truth", "anonymous source", "share this",
    "before media deletes", "too late", "must watch"
]

trusted_sources = [
    "bbc.com",
    "reuters.com",
    "thehindu.com",
    "indianexpress.com",
    "ndtv.com",
    "hindustantimes.com",
    "timesofindia.indiatimes.com",
    "theguardian.com",
    "apnews.com"
]

medium_sources = [
    "news18.com",
    "indiatoday.in",
    "cnbctv18.com",
    "financialexpress.com"
]


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return " ".join(words)


def load_history():
    if not os.path.exists(history_file):
        with open(history_file, "w", encoding="utf-8") as file:
            json.dump([], file)

    with open(history_file, "r", encoding="utf-8") as file:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            return []


def write_history(history):
    with open(history_file, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)


def save_history(record):
    history = load_history()
    history.insert(0, record)
    history = history[:25]
    write_history(history)


def get_domain(url):
    try:
        parsed = urlparse(url)
        return parsed.netloc.replace("www.", "")
    except Exception:
        return ""


def get_source_reliability(domain):
    if not domain:
        return "Not Applicable", 10

    for source in trusted_sources:
        if source in domain:
            return "Trusted Source", 20

    for source in medium_sources:
        if source in domain:
            return "Medium Reliability Source", 12

    return "Unknown Source", 5


def generate_reasons(keyword_count, source_reliability, word_count, confidence, similarity_type, prediction):
    reasons = []

    if keyword_count > 0:
        reasons.append("Suspicious or sensational keywords detected")
    else:
        reasons.append("No major suspicious keywords detected")

    if source_reliability == "Trusted Source":
        reasons.append("Input source belongs to a trusted news domain")
    elif source_reliability == "Medium Reliability Source":
        reasons.append("Input source has medium reliability")
    elif source_reliability == "Unknown Source":
        reasons.append("Input source is not in the trusted source list")

    if word_count < 8:
        reasons.append("Input content is very short, so reliability may reduce")
    elif word_count < 20:
        reasons.append("Input content is moderate in length")
    else:
        reasons.append("Input has sufficient content for deeper analysis")

    if confidence >= 85:
        reasons.append("Machine learning model confidence is high")
    elif confidence >= 65:
        reasons.append("Machine learning model confidence is moderate")
    else:
        reasons.append("Machine learning model confidence is low")

    if similarity_type == "Fake Pattern":
        reasons.append("Input is closer to known fake news language patterns")
    elif similarity_type == "Real Pattern":
        reasons.append("Input is closer to formal real news language patterns")
    else:
        reasons.append("Input pattern is mixed and not strongly similar to one side")

    if prediction == 1:
        reasons.append("Language style appears closer to formal news writing")
    else:
        reasons.append("Language style appears closer to rumor or exaggerated content")

    return reasons


def load_reference_samples():
    try:
        fake_df = pd.read_csv("Fake.csv")
        true_df = pd.read_csv("True.csv")

        fake_samples = fake_df["text"].dropna().astype(str).head(200).tolist()
        true_samples = true_df["text"].dropna().astype(str).head(200).tolist()

        fake_samples = [preprocess_text(x) for x in fake_samples if x.strip()]
        true_samples = [preprocess_text(x) for x in true_samples if x.strip()]

        return fake_samples, true_samples
    except Exception:
        return [], []


fake_reference_samples, real_reference_samples = load_reference_samples()


def compute_similarity_scores(cleaned_text):
    if not fake_reference_samples or not real_reference_samples:
        return 0.0, 0.0, "Mixed Pattern"

    fake_corpus = [cleaned_text] + fake_reference_samples[:50]
    real_corpus = [cleaned_text] + real_reference_samples[:50]

    fake_matrix = vectorizer.transform(fake_corpus)
    real_matrix = vectorizer.transform(real_corpus)

    fake_similarities = cosine_similarity(fake_matrix[0:1], fake_matrix[1:]).flatten()
    real_similarities = cosine_similarity(real_matrix[0:1], real_matrix[1:]).flatten()

    fake_score = round(float(fake_similarities.max()) * 100, 2) if len(fake_similarities) > 0 else 0.0
    real_score = round(float(real_similarities.max()) * 100, 2) if len(real_similarities) > 0 else 0.0

    if fake_score > real_score + 5:
        similarity_type = "Fake Pattern"
    elif real_score > fake_score + 5:
        similarity_type = "Real Pattern"
    else:
        similarity_type = "Mixed Pattern"

    return fake_score, real_score, similarity_type


@app.route("/")
def home():
    return render_template("index.html", model_info=model_info)


@app.route("/history")
def history():
    records = load_history()
    return render_template("history.html", records=records)


@app.route("/about")
def about():
    return render_template("about.html", model_info=model_info)


@app.route("/delete-history/<record_id>", methods=["POST"])
def delete_history(record_id):
    history = load_history()
    updated_history = [item for item in history if item.get("id") != record_id]
    write_history(updated_history)
    return redirect(url_for("history"))


@app.route("/clear-history", methods=["POST"])
def clear_history():
    write_history([])
    return redirect(url_for("history"))


@app.route("/download-report")
def download_report():
    report = session.get("latest_report")

    if not report:
        return redirect(url_for("home"))

    file_path = "analysis_report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=30)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=20,
        textColor=colors.HexColor("#1f3c88"),
        spaceAfter=14
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.white,
        backColor=colors.HexColor("#4f46e5"),
        leftIndent=0,
        spaceBefore=10,
        spaceAfter=8,
        borderPadding=6
    )

    normal_style = styles["Normal"]
    normal_style.fontSize = 10
    normal_style.leading = 14

    story = []

    story.append(Paragraph("📰 Fake News Detector", title_style))
    story.append(Paragraph("Professional News Analysis Report", styles["Heading3"]))
    story.append(Spacer(1, 12))

    result_color = "#16a34a"
    if "Suspicious" in str(report.get("result", "")) or "Fake" in str(report.get("result", "")):
        result_color = "#dc2626"
    elif "Moderately" in str(report.get("result", "")):
        result_color = "#ca8a04"

    result_box = Table(
        [[Paragraph(f"<b>Final Result:</b> <font color='{result_color}'>{report.get('result', '-')}</font>", normal_style)]],
        colWidths=[500]
    )
    result_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef2ff")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#c7d2fe")),
        ("PADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(result_box)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Basic Information", heading_style))
    basic_data = [
        ["Date & Time", str(report.get("timestamp", "-"))],
        ["Confidence", f"{report.get('confidence', '-')}%"],
        ["Credibility Score", f"{report.get('credibility_score', '-')}/100"],
        ["Risk Level", str(report.get("risk", "-"))],
        ["Input Type", str(report.get("source", "-"))],
        ["Source Reliability", str(report.get("source_reliability", "-"))],
        ["Domain", str(report.get("domain", "-"))],
        ["Word Count", str(report.get("word_count", "-"))],
    ]
    basic_table = Table(basic_data, colWidths=[150, 350])
    basic_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8fafc")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(basic_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Analysis Summary", heading_style))
    story.append(Paragraph(f"<b>Main Reason:</b> {report.get('reason', '-')}", normal_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Summary:</b> {report.get('summary', '-')}", normal_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Similarity Analysis", heading_style))
    similarity_data = [
        ["Fake Pattern Similarity", f"{report.get('fake_similarity', '-')}%"],
        ["Real Pattern Similarity", f"{report.get('real_similarity', '-')}%"],
        ["Closest Pattern", str(report.get("similarity_type", "-"))],
    ]
    sim_table = Table(similarity_data, colWidths=[180, 320])
    sim_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8fafc")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(sim_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Suspicious Keywords", heading_style))
    keywords_text = ", ".join(report.get("keywords", [])) if report.get("keywords") else "None"
    story.append(Paragraph(keywords_text, normal_style))
    story.append(Spacer(1, 12))

    reason_list = report.get("reason_list", [])
    if reason_list:
        story.append(Paragraph("Detailed Reasons", heading_style))
        for item in reason_list:
            story.append(Paragraph(f"• {item}", normal_style))
            story.append(Spacer(1, 4))

    doc.build(story)
    return send_file(file_path, as_attachment=True)


@app.route("/predict", methods=["POST"])
def predict():
    text = ""
    raw_text = ""
    url = request.form.get("url", "").strip()
    news = request.form.get("news", "").strip()

    if url:
        source_input_type = "URL Input"
        domain = get_domain(url)
        source_reliability, source_score = get_source_reliability(domain)

        if NEWSPAPER_AVAILABLE:
            try:
                article = Article(url)
                article.download()
                article.parse()
                text = article.text.strip()
                raw_text = text

                if not text:
                    text = news
                    raw_text = news
            except Exception:
                text = news
                raw_text = news
        else:
            text = news
            raw_text = news
    else:
        source_input_type = "Manual Text"
        domain = ""
        source_reliability, source_score = "Not Applicable", 10
        text = news
        raw_text = news

    if not text:
        return render_template(
            "index.html",
            prediction_text="⚠️ Please enter news text or paste a news URL",
            model_info=model_info
        )

    if len(text.strip()) < 20:
        return render_template(
            "index.html",
            prediction_text="⚠️ Enter more detailed news",
            model_info=model_info
        )

    cleaned_text = preprocess_text(text)
    transformed = vectorizer.transform([cleaned_text])

    prediction = model.predict(transformed)[0]
    probability = model.predict_proba(transformed)[0]
    confidence = round(max(probability) * 100, 2)

    lower_text = raw_text.lower()
    matched_keywords = [word for word in fake_keywords if word in lower_text]
    keyword_count = len(matched_keywords)
    word_count = len(raw_text.split())

    fake_similarity, real_similarity, similarity_type = compute_similarity_scores(cleaned_text)

    ml_score = round(confidence * 0.5, 2)
    similarity_score = min(real_similarity * 0.2, 20)
    fake_similarity_penalty = min(fake_similarity * 0.15, 15)
    keyword_penalty = keyword_count * 6
    sensational_penalty = 0

    if raw_text.count("!") >= 2:
        sensational_penalty += 5

    if raw_text.isupper():
        sensational_penalty += 5

    if word_count >= 20:
        content_score = 10
    elif word_count >= 10:
        content_score = 6
    else:
        content_score = 2

    credibility_score = int(
        ml_score + source_score + similarity_score + content_score
        - fake_similarity_penalty - keyword_penalty - sensational_penalty
    )

    credibility_score = max(0, min(100, credibility_score))

    if credibility_score >= 75:
        final_result = "Highly Reliable News ✅"
        risk = "Low Risk 🟢"
    elif credibility_score >= 55:
        final_result = "Moderately Reliable News 🟡"
        risk = "Medium Risk 🟡"
    elif credibility_score >= 35:
        final_result = "Suspicious News ⚠️"
        risk = "High Risk 🟠"
    else:
        final_result = "Highly Suspicious News ❌"
        risk = "Very High Risk 🔴"

    reasons = generate_reasons(
        keyword_count=keyword_count,
        source_reliability=source_reliability,
        word_count=word_count,
        confidence=confidence,
        similarity_type=similarity_type,
        prediction=prediction
    )



    summary_words = raw_text.split()[:25]
    summary = " ".join(summary_words)
    if len(raw_text.split()) > 25:
        summary += "..."

    if word_count < 8:
        extra_note = "Very short input. Result may be less accurate."
    else:
        extra_note = "Sufficient content available for analysis."

    preview = " ".join(raw_text.split()[:12])
    if len(raw_text.split()) > 12:
        preview += "..."

    record = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().strftime("%d-%m-%Y %I:%M %p"),
        "input_type": source_input_type,
        "domain": domain if domain else "-",
        "result": final_result,
        "confidence": confidence,
        "credibility_score": credibility_score,
        "risk": risk,
        "source_reliability": source_reliability,
        "preview": preview
    }

    save_history(record)

    
    return render_template(
        "index.html",
        prediction_text=final_result,
        confidence=confidence,
        credibility_score=credibility_score,
        risk=risk,
        reason=primary_reason,
        reason_list=reasons,
        keywords=matched_keywords,
        source=source_input_type,
        source_reliability=source_reliability,
        domain=domain,
        extra_note=extra_note,
        word_count=word_count,
        summary=summary,
        fake_similarity=fake_similarity,
        real_similarity=real_similarity,
        similarity_type=similarity_type,
        model_info=model_info,
        report_available=false
    )


if __name__ == "__main__":
    app.run(debug=True)