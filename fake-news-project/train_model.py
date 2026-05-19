import pickle
import random
import pandas as pd



random.seed(42)


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return " ".join(words)


real_topics = [
    "government announces new education policy for schools",
    "india wins cricket series after strong team performance",
    "scientists discover new renewable energy method",
    "health ministry releases vaccination update for all states",
    "supreme court gives decision in public interest case",
    "weather department warns of heavy rainfall in coastal districts",
    "university publishes official examination schedule",
    "police begin investigation in robbery case",
    "metro line to open next month for public transport",
    "researchers develop artificial intelligence system for healthcare",
    "stock market shows steady growth in trading session",
    "official report highlights economic reforms and employment plans",
    "city administration launches road safety campaign",
    "technology company announces smartphone launch this week",
    "railway department issues new travel advisory for passengers",
    "bank releases notice about digital payment security",
    "district hospital starts new emergency care unit",
    "space agency confirms successful satellite mission",
    "election commission issues official voting guidelines",
    "environment ministry starts clean river program"
]

fake_topics = [
    "shocking miracle cure discovered doctors hate this trick",
    "100 percent guaranteed way to earn money fast from home",
    "click here to become rich overnight without effort",
    "unbelievable secret revealed that will change your life forever",
    "viral message claims free money for everyone instantly",
    "secret government plan leaked on social media",
    "one trick can make you millionaire in one day",
    "miracle pill cures every disease instantly guaranteed",
    "breaking viral post says earth will go dark tomorrow",
    "you wont believe what happened next shocking secret revealed",
    "free laptop scheme available for everyone without registration",
    "celebrity death rumor goes viral before official confirmation",
    "hidden formula promises instant job placement with no interview",
    "magic investment plan doubles money in one hour",
    "spam post claims bank account upgrade gives free cashback forever",
    "viral image claims aliens landed in major city",
    "instant weight loss solution works in five minutes",
    "secret exam paper leaked to all students online",
    "mysterious post says mobile towers spread dangerous waves tomorrow",
    "fake recruitment message promises government job without exam"
]

real_prefixes = [
    "official statement says",
    "according to government report",
    "news agencies reported that",
    "official sources confirmed that",
    "the ministry announced that",
    "authorities stated that",
    "a verified report says",
    "media briefing confirmed that"
]

fake_prefixes = [
    "viral post says",
    "social media message claims",
    "anonymous source reveals",
    "forwarded message says",
    "shocking update says",
    "secret post claims",
    "unverified message says",
    "rumor online says"
]

real_suffixes = [
    "with detailed information released to the public",
    "after an official press conference",
    "based on verified documents and statements",
    "as confirmed by authorities",
    "with data shared in the official report",
    "for public awareness and transparency",
    "during a scheduled media briefing",
    "after review by the concerned department"
]

fake_suffixes = [
    "share this with everyone immediately",
    "before media deletes the truth",
    "act now before it is too late",
    "this is hidden from the public",
    "do not ignore this shocking news",
    "everyone is talking about this secret",
    "watch before it disappears",
    "this can change your life instantly"
]


def build_dataset():
    rows = []

    # Real samples
    for topic in real_topics:
        rows.append({"text": topic, "label": 1})
        for prefix in real_prefixes:
            rows.append({"text": f"{prefix} {topic}", "label": 1})
        for suffix in real_suffixes:
            rows.append({"text": f"{topic} {suffix}", "label": 1})

    # Fake samples
    for topic in fake_topics:
        rows.append({"text": topic, "label": 0})
        for prefix in fake_prefixes:
            rows.append({"text": f"{prefix} {topic}", "label": 0})
        for suffix in fake_suffixes:
            rows.append({"text": f"{topic} {suffix}", "label": 0})

    # Extra mixed real samples
    extra_real = [
        "official update from health department confirms vaccination progress across districts",
        "verified news report says rescue teams reached the earthquake affected area",
        "police confirmed that the investigation is ongoing after the complaint was filed",
        "the government released an official statement on fuel price revision",
        "the university administration announced revised exam dates on its official website",
        "scientists published peer reviewed findings on renewable energy storage",
        "transport department launched a safety awareness program for drivers",
        "the bank warned users against fraud through a public advisory",
        "official election guidelines were published for first time voters",
        "the court hearing was postponed according to the official notice"
    ]

    extra_fake = [
        "viral post says free government jobs are available without exam or interview",
        "miracle herb cures diabetes cancer and heart disease instantly",
        "social media rumor claims all banks will shut down tomorrow morning",
        "forwarded message says students will get free laptops without registration",
        "shocking message says internet will stop working nationwide tonight",
        "secret formula can make you rich without investment guaranteed",
        "unbelievable trick reveals how to get unlimited money instantly",
        "spam message promises high salary jobs with zero qualification",
        "viral claim says moon will disappear for three days next week",
        "mystery video proves a hidden plan that media will never show"
    ]

    for text in extra_real:
        rows.append({"text": text, "label": 1})

    for text in extra_fake:
        rows.append({"text": text, "label": 0})

    random.shuffle(rows)
    return pd.DataFrame(rows)


df = build_dataset()
df["clean_text"] = df["text"].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Multinomial Naive Bayes": MultinomialNB()
}

best_model = None
best_model_name = ""
best_accuracy = 0.0
best_report = ""

for model_name, model in models.items():
    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)
    acc = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=["Fake", "Real"])

    print(f"\nModel: {model_name}")
    print(f"Accuracy: {round(acc * 100, 2)}%")
    print(report)

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = model_name
        best_report = report

with open("model.pkl", "wb") as model_file:
    pickle.dump(best_model, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

model_info = {
    "best_model": best_model_name,
    "accuracy": round(best_accuracy * 100, 2),
    "dataset_size": int(len(df)),
    "real_samples": int((df["label"] == 1).sum()),
    "fake_samples": int((df["label"] == 0).sum()),
    "classification_report": best_report
}

with open("model_info.json", "w") as info_file:
    json.dump(model_info, info_file, indent=4)

print("\nBest model selected and saved successfully!")
print(f"Best Model: {best_model_name}")
print(f"Final Accuracy: {round(best_accuracy * 100, 2)}%")
print(f"Dataset Size: {len(df)}")