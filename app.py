import streamlit as st
import numpy as np
import pickle
import json
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from nltk.translate.bleu_score import corpus_bleu
import nltk
nltk.download('punkt', quiet=True)

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VisionWrite",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ VisionWrite — Automatic Image Caption Generator")
st.markdown("Upload any image and get an AI-generated English caption using CNN + LSTM.")
st.divider()

# ── Load all assets once (cached) ─────────────────────────────────────────────
@st.cache_resource
def load_assets():
    import gdown, os
    os.makedirs("model", exist_ok=True)

    if not os.path.exists("model/caption_model.keras.h5"):
        with st.spinner("📥 Downloading model... please wait"):
            gdown.download(
                "https://drive.google.com/uc?id=1dm8mN0bpro4Tn6HkVn7TIAGBuA4kdbVR",
                "model/caption_model.keras.h5",
                quiet=False
            )

    caption_model = load_model("model/caption_model.keras.h5")

    with open("model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("model/model_config.json", "r") as f:
        config = json.load(f)

    inception_v3_model = InceptionV3(weights='imagenet', input_shape=(299, 299, 3))
    inception_v3_model.layers.pop()
    feature_extractor = Model(
        inputs=inception_v3_model.inputs,
        outputs=inception_v3_model.layers[-2].output
    )
    return caption_model, tokenizer, config, feature_extractor

caption_model, tokenizer, config, feature_extractor = load_assets()

MAX_LEN    = config["max_caption_length"]
VOCAB_SIZE = config["vocab_size"]
CNN_DIM    = config["cnn_output_dim"]   # 2048

# ── Feature extraction — matches your Cell 9 exactly ─────────────────────────
def extract_features_from_pil(image: Image.Image) -> np.ndarray:
    img = image.resize((299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = feature_extractor.predict(img, verbose=0)
    return features.flatten()

# ── Greedy Search — matches your Cell 17 exactly ─────────────────────────────
def greedy_generator(image_features):
    in_sequence = [tokenizer.word_index['start']]

    for i in range(MAX_LEN):
        # padding='post'  ← exactly as in your notebook
        padded_sequence = pad_sequences(
            [in_sequence], maxlen=MAX_LEN, padding='post'
        )
        image_input  = image_features.reshape((1, -1)).astype(np.float32)
        seq_input    = padded_sequence.astype(np.int32)

        prediction       = caption_model.predict([image_input, seq_input], verbose=0)
        predicted_word_id = np.argmax(prediction)

        if predicted_word_id >= VOCAB_SIZE:
            break

        in_sequence.append(predicted_word_id)
        predicted_word = tokenizer.index_word.get(predicted_word_id, '')

        if predicted_word == 'end':
            break

    final_caption = ' '.join([
        tokenizer.index_word[i]
        for i in in_sequence[1:]
        if i in tokenizer.index_word and tokenizer.index_word[i] != 'end'
    ])
    return final_caption

# ── Beam Search — matches your Cell 18 exactly ───────────────────────────────
def beam_search_generator(image_features, K_beams=3, log=True):
    start_token_id = tokenizer.word_index['start']
    start_word     = [[[start_token_id], 0.0]]

    image_input = image_features.reshape((1, -1)).astype(np.float32)

    while len(start_word[0][0]) < MAX_LEN:
        temp = []
        for s in start_word:
            # padding='post' ← exactly as in your notebook
            sequence       = pad_sequences([s[0]], maxlen=MAX_LEN, padding='post')
            sequence_input = sequence.astype(np.int32)

            preds      = caption_model.predict([image_input, sequence_input], verbose=0)
            word_preds = np.argsort(preds[0])[-K_beams:]

            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += np.log(preds[0][w] + 1e-9) if log else preds[0][w]
                temp.append([next_cap, prob])

        start_word = temp
        start_word = sorted(start_word, reverse=True, key=lambda l: l[1])
        start_word = start_word[:K_beams]

        if start_word[0][0][-1] == tokenizer.word_index.get('end', -1):
            break

    best_sequence = start_word[0][0]
    final_caption = ' '.join([
        tokenizer.index_word[i]
        for i in best_sequence[1:]
        if i in tokenizer.index_word and tokenizer.index_word[i] != 'end'
    ])
    return final_caption

# ── BLEU Score — matches your Cell 19 signature exactly ──────────────────────
def compute_bleu(actual_captions, greedy_caption, beam_caption):
    references  = [cap.split() for cap in actual_captions]
    greedy_hyp  = greedy_caption.split()
    beam_hyp    = beam_caption.split()

    score_greedy_4 = corpus_bleu([references], [greedy_hyp], weights=(0.25, 0.25, 0.25, 0.25))
    score_greedy_2 = corpus_bleu([references], [greedy_hyp], weights=(0.5, 0.5, 0, 0))
    score_bs_4     = corpus_bleu([references], [beam_hyp],   weights=(0.25, 0.25, 0.25, 0.25))
    score_bs_2     = corpus_bleu([references], [beam_hyp],   weights=(0.5, 0.5, 0, 0))

    return {
        "greedy_bleu4": round(score_greedy_4, 4),
        "greedy_bleu2": round(score_greedy_2, 4),
        "beam_bleu4":   round(score_bs_4, 4),
        "beam_bleu2":   round(score_bs_2, 4),
    }

# ── Sidebar settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    method = st.radio(
        "Caption Method",
        ["Both (Greedy + Beam)", "Greedy Search Only", "Beam Search Only"],
        index=0
    )
    k_beams = st.slider("Beam Width (K)", min_value=2, max_value=5, value=3)
    show_bleu = st.checkbox("Show BLEU Scores", value=True)
    st.divider()
    st.caption("Model: InceptionV3 + LSTM")
    st.caption("Dataset: Flickr8K")
    st.caption("Course: CSE423 — NLP")

# ── Main upload UI ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📤 Upload an image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("🔍 Extracting image features via InceptionV3..."):
            features = extract_features_from_pil(image)
        st.success("Features extracted!")

        if st.button("✨ Generate Caption", use_container_width=True):

            greedy_cap, beam_cap = None, None

            if method in ["Both (Greedy + Beam)", "Greedy Search Only"]:
                with st.spinner("✍️ Running Greedy Search..."):
                    greedy_cap = greedy_generator(features)

            if method in ["Both (Greedy + Beam)", "Beam Search Only"]:
                with st.spinner(f"🔎 Running Beam Search (K={k_beams})..."):
                    beam_cap = beam_search_generator(features, K_beams=k_beams)

            st.divider()

            if greedy_cap:
                st.markdown("### 🟢 Greedy Search")
                st.info(f"**{greedy_cap}**")

            if beam_cap:
                st.markdown("### 🔵 Beam Search")
                st.success(f"**{beam_cap}**")

            # BLEU scores (only if both captions available)
            if show_bleu and greedy_cap and beam_cap:
                st.divider()
                st.markdown("### 📊 BLEU Evaluation")
                st.caption("Note: BLEU scores here compare the two generated captions against each other as proxy references.")

                proxy_refs = [greedy_cap, beam_cap]
                scores = compute_bleu(proxy_refs, greedy_cap, beam_cap)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Greedy BLEU-4", scores["greedy_bleu4"])
                c2.metric("Greedy BLEU-2", scores["greedy_bleu2"])
                c3.metric("Beam BLEU-4",   scores["beam_bleu4"])
                c4.metric("Beam BLEU-2",   scores["beam_bleu2"])

st.divider()
st.caption("VisionWrite | CSE423 NLP | SRM University-AP | May 2026")