import joblib
import gradio as gr
from pathlib import Path

MODEL_PATHS = [
    Path("artifacts/spam_nb_tfidf_stem.joblib"),
    Path("artifacts/spam_nb_tfidf.joblib")
]

def load_model():
    for p in MODEL_PATHS:
        if p.exists():
            return joblib.load(p)
    raise FileNotFoundError("No saved model found in artifacts/")

model = load_model()

def predict_spam(text):
    proba = getattr(model, "predict_proba", None)
    if proba is not None:
        p = proba([text])[0][1]
    else:
        # fallback if model lacks predict_proba
        p = float(model.decision_function([text])[0])  # not typical for NB
    label = model.predict([text])[0]
    label_text = "SPAM" if label == 1 else "HAM"
    return { "Prediction": label_text, "Spam probability": float(round(p, 3)) }

demo = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(lines=4, placeholder="Paste an email/SMS message..."),
    outputs=gr.JSON(),
    title="Spam Classifier",
    description="Type a message to check if it's likely Spam or Ham."
)

if __name__ == "__main__":
    import gradio as gr
    
    demo.launch(
        server_name="127.0.0.1",  # ensures it runs locally
        server_port=7860,         # fixed port
        share=True                # also gives a public link
    )
import gradio as gr

def classify_message(message):
    # your spam detection logic here
    return "Spam", 0.85

demo = gr.Interface(
    fn=classify_message,
    inputs="text",
    outputs=["text", "number"],
    flagging_dir="flagged_data"  # folder where CSV will be saved
)


import joblib, gradio as gr, json
model = joblib.load("artifacts/spam_nb_tfidf_stem.joblib")

def predict_spam(text: str):
    proba = model.predict_proba([text])[0][1]
    label = "spam" if proba >= 0.5 else "ham"
    return {"label": label, "prob": float(proba)}

demo = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(lines=4, label="Message"),
    outputs=gr.JSON(label="Result"),
    title="Spam Classifier API",
)

