import re
from pathlib import Path

import requests
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCTC,
    Wav2Vec2Processor,
)

from easyaligner.data.datamodel import SpeechSegment
from easyaligner.data.utils import read_json
from easyaligner.pipelines import pipeline
from easyaligner.text import (
    fuzzy_match,
    load_tokenizer,
    text_normalizer,
)
from easyaligner.vad.pyannote import load_vad_model
from easytranscriber.pipelines import pipeline as transcription_pipeline
from easytranscriber.text.normalization import text_normalizer as easytranscriber_text_normalizer

# --- Chapter II text (needle) ---
url = "https://www.gutenberg.org/cache/epub/98/pg98.txt"
full_text = requests.get(url).text

match = re.search(
    r"(?<=CHAPTER II\.\r\nThe Mail\r\n)[\s\S]+?(?=CHAPTER III\.)",
    full_text,
)
text = match.group().strip()

# --- Download audio ---
tutorial_dir = Path("data/tutorials")
filepath_pattern = "tale-of-two-cities_long-en/taleoftwocities_01_dickens_128kb.mp3"

snapshot_download(
    "Lauler/easytranscriber_tutorials",
    repo_type="dataset",
    local_dir=tutorial_dir,
    allow_patterns=filepath_pattern,
)

filepath = Path(tutorial_dir) / filepath_pattern
audio_dir = filepath.parent
audio_files = [filepath.name]

# --- Step 1: Transcribe the full audio file ---
# This runs VAD -> ASR -> emission extraction -> forced alignment on the entire file,
# giving us word-level timestamps for everything spoken.
tokenizer = load_tokenizer(language="english")

transcription_pipeline(
    vad_model="pyannote",
    emissions_model="facebook/wav2vec2-base-960h",
    transcription_model="distil-whisper/distil-large-v3.5",
    audio_paths=audio_files,
    audio_dir=str(audio_dir),
    backend="ct2",
    language="en",
    tokenizer=tokenizer,
    text_normalizer_fn=easytranscriber_text_normalizer,
    cache_dir="models",
)

# --- Step 2: Fuzzy match Chapter II in the transcribed output ---
alignment_json = Path("output/alignments") / filepath.with_suffix(".json").name
audio_meta = read_json(alignment_json)

match = fuzzy_match(needle=text, haystack=audio_meta.speeches)

if match is None:
    raise RuntimeError(
        "Could not find Chapter II in the transcription. Try lowering the threshold."
    )

print(f"Fuzzy match score: {match.score:.1f}")
print(f"Found Chapter II: {match.start:.1f}s – {match.end:.1f}s")

# --- Step 3: Forced alignment with the ground-truth text and discovered timestamps ---
span_list = list(tokenizer.span_tokenize(text))

speeches = [
    [
        SpeechSegment(
            speech_id="chapter-ii",
            text=text,
            text_spans=span_list,
            start=match.start,
            end=match.end,
        )
    ]
]

model_vad = load_vad_model()
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h").to("cuda").half()
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

pipeline(
    vad_model=model_vad,
    emissions_model=model,
    processor=processor,
    audio_paths=audio_files,
    audio_dir=audio_dir,
    speeches=speeches,
    alignment_strategy="speech",
    text_normalizer_fn=text_normalizer,
    tokenizer=tokenizer,
    start_wildcard=True,
    end_wildcard=True,
    blank_id=processor.tokenizer.pad_token_id,
    word_boundary="|",
)
