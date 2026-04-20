import streamlit as st
import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# WEB APP FOR CAPSTONE PROJECT
# I built this to summarize long news articles using the bart model i fine tuned

st.set_page_config(page_title="AI News Summarizer", page_icon="🤖", layout="centered")

# Check if I can use GPU/Mac MPS for speed, otherwise fallback to CPU (Streamlit Cloud uses CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# @st.cache_resource is super important here so I only load the heavy model once.
# Otherwise, it reloads on every button click and crashes the cloud app.
@st.cache_resource
def load_summarizer():
    model_id = "MazinB04/FINETUNED_BART"
    tokenizer_id = "facebook/bart-large-cnn"

    # Grab my huggingface token from streamlit secrets (cloud) or local env
    try:
        token = st.secrets["HF_TOKEN"]
    except Exception:
        token = os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, token=token)

    # low_cpu_mem_usage stops RAM spikes during cloud deployment
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        token=token,
        low_cpu_mem_usage=True
    ).to(DEVICE)

    return tokenizer, model


tokenizer, model = load_summarizer()

# Streamlit resets variables on every interaction, so I need session state to remember inputs
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""
if 'final_summary' not in st.session_state:
    st.session_state.final_summary = None

st.title("🤖 Automatic News Summarizer")
st.write(f"Leverage advanced Natural Language Processing to summarise news articles. *(Hardware: {DEVICE.upper()})*")

st.subheader("Select a Topic or Input Custom Text")

# Some preset articles
football_text = "Wrexham's push for the League One play-offs continued with a hard-fought 1-0 victory against Mansfield Town at the Racecourse Ground. The match, played in front of a sold-out crowd, was decided by a clinical strike from star forward Paul Mullin in the 64th minute. Mansfield had dominated possession for large periods of the first half, testing the Wrexham defense with several well-timed crosses, but were unable to find a breakthrough. Wrexham goalkeeper Arthur Okonkwo made two vital saves to keep the score level before the break. In the second half, the hosts looked more dangerous on the counter-attack. Mullin's goal came after a swift transition, where he latched onto a through ball and fired low into the bottom corner. Mansfield pushed for an equalizer in the closing stages, hitting the post in injury time, but Wrexham held on for a vital three points. The win moves the Welsh club up to third in the table, while Mansfield remain in the hunt for a top-six spot despite the narrow defeat."
space_text = "NASA’s Artemis II mission is currently preparing for a historic flight that will send four astronauts around the Moon and back to Earth. This mission marks the first time humans will visit the lunar vicinity since the Apollo 17 mission in 1972. The crew consists of Commander Reid Wiseman, Pilot Victor Glover, and Mission Specialists Christina Koch and Jeremy Hansen. Artemis II is a 10-day flight test designed to verify the Space Launch System (SLS) rocket and the Orion spacecraft’s life-support systems for crewed operations. Unlike later missions, Artemis II will not land on the surface; instead, it will perform a 'free-return trajectory,' using the Moon’s gravity to slingshot the spacecraft back toward Earth. This mission is a critical stepping stone for Artemis III, which aims to land the first woman and first person of color on the lunar surface."

col_a, col_b, col_c = st.columns(3)
if col_a.button("⚽ Football "):
    st.session_state.text_input = football_text
if col_b.button("🚀 Space Mission"):
    st.session_state.text_input = space_text
if col_c.button("🧹 Clear All"):
    st.session_state.text_input = ""
    st.session_state.final_summary = None

# Input field linked directly to my session state
article_text = st.text_area("Article Content:", height=200, key="text_input",
                            placeholder="Enter article text here...")

st.sidebar.header("Summarization Controls")
max_len = st.sidebar.slider("Summary Max Length", 30, 300, 100)
# Beam search gives better summaries but takes a bit longer to compute
num_beams = st.sidebar.slider("Model Accuracy (Beams)", 1, 5, 4)

if st.button("✨ Generate Summary", type="primary"):
    if article_text.strip():
        input_word_count = len(article_text.split())

        with st.spinner(f"Summarising {input_word_count} words..."):
            try:
                # BART has a 1024 token limit. If my article is too long,
                # I split it into 600-word chunks, summarize each, and combine them at the end.
                words = article_text.split()
                chunk_size = 600
                chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

                all_summaries = []
                for chunk in chunks:
                    inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True).to(DEVICE)
                    summary_ids = model.generate(inputs["input_ids"], num_beams=num_beams, max_length=max_len)
                    all_summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))

                # Do a final summary pass if there were multiple chunks
                if len(chunks) > 1:
                    final_combined_text = " ".join(all_summaries)
                    inputs = tokenizer(final_combined_text, return_tensors="pt", max_length=1024, truncation=True).to(
                        DEVICE)
                    summary_ids = model.generate(inputs["input_ids"], num_beams=num_beams, max_length=max_len)
                    st.session_state.final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                else:
                    st.session_state.final_summary = all_summaries[0]

                st.session_state.original_count = input_word_count

            except Exception as e:
                st.error(f"Processing error: {e}")
    else:
        st.warning("Please enter an article to summarise.")

# Display results if I have a summary
if st.session_state.get('final_summary'):
    st.divider()
    st.subheader("🎯 Summary Output")
    st.success(st.session_state.final_summary)

    # Calculate how much I compressed the text
    summary_words = len(st.session_state.final_summary.split())
    original_words = st.session_state.original_count
    reduction = round((1 - (summary_words / original_words)) * 100) if original_words > summary_words else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("Input Words", original_words)
    m2.metric("Summary Words", summary_words)
    m3.metric("Reduction", f"{reduction}%")
