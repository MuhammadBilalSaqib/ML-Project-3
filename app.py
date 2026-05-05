import re
import string
import json
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

app = Flask(__name__)

# Global variable for the pipeline
qa_pipeline = None

def get_qa_pipeline():
    global qa_pipeline
    if qa_pipeline is None:
        print("Loading QA model...")
        qa_pipeline = pipeline('question-answering', model="deepset/roberta-base-squad2")
        print("Model loaded successfully!")
    return qa_pipeline


# ── Exact Match helper (standard SQuAD metric) ───────────────────────────────

def normalize(text):
    """Lowercase, strip articles, punctuation, and extra whitespace."""
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    return ' '.join(text.split())

def exact_match(prediction, gold_answers):
    """Return True if prediction matches any gold answer after normalization."""
    pred_norm = normalize(prediction)
    return any(normalize(g) == pred_norm for g in gold_answers)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    context = data.get("context", "").strip()
    question = data.get("question", "").strip()

    if not context or not question:
        return jsonify({"error": "Both context and question are required."}), 400

    try:
        pipeline = get_qa_pipeline()
        result = pipeline(question=question, context=context)

        return jsonify({
            "answer": result["answer"],
            "confidence": round(result["score"] * 100, 2),
            "start": result["start"],
            "end": result["end"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/evaluate", methods=["GET"])
def evaluate():
    def generate():
        try:
            dataset = load_dataset("squad_v2", split="validation[:20]")
            total = len(dataset)

            em_scores = []
            true_answerability = []
            pred_answerability = []

            pipeline = get_qa_pipeline()

            for i, item in enumerate(dataset):
                context = item["context"]
                question = item["question"]
                gold_answers = item["answers"]["text"]
                is_answerable = len(gold_answers) > 0

                result = pipeline(question=question, context=context)
                predicted_answer = result["answer"].strip()

                true_answerability.append(1 if is_answerable else 0)
                pred_answerability.append(1 if predicted_answer != "" else 0)

                if is_answerable:
                    em_scores.append(1 if exact_match(predicted_answer, gold_answers) else 0)

                # Stream progress after each sample
                progress = round((i + 1) / total * 100)
                yield f"data: {json.dumps({'progress': progress, 'current': i + 1, 'total': total})}\n\n"
                import sys; sys.stdout.flush()

            # Stream final results
            em_accuracy = round(sum(em_scores) / len(em_scores) * 100, 2) if em_scores else 0
            overall_accuracy = round(accuracy_score(true_answerability, pred_answerability) * 100, 2)
            cm = confusion_matrix(true_answerability, pred_answerability, labels=[0, 1]).tolist()

            yield f"data: {json.dumps({'done': True, 'exact_match_accuracy': em_accuracy, 'answerability_accuracy': overall_accuracy, 'total_samples': total, 'confusion_matrix': {'labels': ['Unanswerable', 'Answerable'], 'matrix': cm}})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
