"""
OmniHCI Studio - Tab 5: NLP & Text Intelligence Suite
"""
import gradio as gr
from src.nlp.nltk_tools import run_nltk_pipeline
from src.nlp.classifier import IntentSentimentClassifier
from src.nlp.llm_nlp import OllamaNLP

def create_nlp_tab():
    classifier = IntentSentimentClassifier()
    llm_nlp = OllamaNLP()

    with gr.Tab("🧠 NLP & Text Intelligence"):
        gr.Markdown(
            "### 📖 Natural Language Processing & Text Understanding\n"
            "Analyze syntactic structures, train/evaluate statistical intent classifiers, and leverage local LLMs for semantic extraction."
        )

        with gr.Tabs():
            # Sub-Tab 1: NLTK
            with gr.Tab("🔬 Linguistic Analysis Pipeline"):
                with gr.Row():
                    with gr.Column(scale=1):
                        nltk_text = gr.Textbox(
                            label="Input Text for Linguistic Processing",
                            lines=4,
                            value="Human-Computer Interaction systems seamlessly combine computer vision, speech signal processing, and natural language understanding to create intuitive interfaces."
                        )
                        nltk_btn = gr.Button("🔍 Run Syntactic Analysis", variant="primary")
                    with gr.Column(scale=1):
                        nltk_stats = gr.Markdown("Linguistic summary will appear here.")
                        nltk_json = gr.JSON(label="Detailed Parse Breakdown (Tokens, POS, Entities, Lemmas)")

                def on_nltk(text):
                    if not text:
                        return "Please enter text.", {}
                    res = run_nltk_pipeline(text)
                    stats = (
                        f"**Sentences:** `{res['sentence_count']}` | "
                        f"**Tokens:** `{res['word_count']}` | "
                        f"**Clean Content Words:** `{res['clean_word_count']}`\n\n"
                        f"**Lexical Diversity Index:** `{res['lexical_diversity']}`\n\n"
                        f"**Prominent Keywords:** {', '.join([f'{w} ({c})' for w, c in res['top_frequencies'][:5]])}"
                    )
                    return stats, res

                nltk_btn.click(on_nltk, inputs=[nltk_text], outputs=[nltk_stats, nltk_json])

            # Sub-Tab 2: Classifier
            with gr.Tab("🎯 Statistical Intent Classifier"):
                with gr.Row():
                    with gr.Column(scale=1):
                        clf_input = gr.Textbox(
                            label="User Statement / Command",
                            placeholder="Type a query, e.g., 'Can you detect facial fatigue and track hand gestures in this frame?'",
                            lines=2
                        )
                        clf_btn = gr.Button("🏷️ Classify Intent", variant="primary")
                        clf_res = gr.Markdown("Classification result will appear here.")
                    with gr.Column(scale=1):
                        gr.Markdown("#### Model Evaluation Metrics")
                        eval_info = gr.Markdown(
                            f"**Validation Accuracy:** `{classifier.accuracy}%`\n\n"
                            f"**Classes Covered:** `{', '.join(classifier.get_metrics()['classes'])}`\n\n"
                            f"```\n{classifier.report_text}\n```"
                        )

                def on_classify(txt):
                    if not txt:
                        return "Please provide input text."
                    label, conf = classifier.predict(txt)
                    return f"### Predicted Intent: `{label.upper()}`\n**Confidence Score:** `{conf}%`"

                clf_btn.click(on_classify, inputs=[clf_input], outputs=[clf_res])

            # Sub-Tab 3: LLM Zero-Shot
            with gr.Tab("🤖 LLM Zero-Shot Intelligence"):
                with gr.Row():
                    with gr.Column(scale=1):
                        llm_input = gr.Textbox(
                            label="Input Text for Semantic Reasoning",
                            lines=3,
                            value="The real-time camera feed exhibited low latency and high accuracy, but the ambient acoustic noise occasionally interfered with voice transcription."
                        )
                        with gr.Row():
                            senti_btn = gr.Button("❤️ Sentiment")
                            intent_btn = gr.Button("🎯 Intent")
                            ner_btn = gr.Button("🏷️ Entities")
                            sum_btn = gr.Button("📝 Summarize")

                    with gr.Column(scale=1):
                        llm_output = gr.Textbox(label="LLM Analytical Output", lines=6, interactive=False)

                senti_btn.click(lambda t: f"Sentiment: {llm_nlp.detect_sentiment(t)}", inputs=[llm_input], outputs=[llm_output])
                intent_btn.click(lambda t: f"Intent: {llm_nlp.classify_intent(t)}", inputs=[llm_input], outputs=[llm_output])
                ner_btn.click(lambda t: llm_nlp.extract_entities(t), inputs=[llm_input], outputs=[llm_output])
                sum_btn.click(lambda t: llm_nlp.summarize(t), inputs=[llm_input], outputs=[llm_output])
