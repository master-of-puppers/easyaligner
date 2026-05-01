import re
from pathlib import Path

import requests
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCTC,
    Wav2Vec2Processor,
)

from easyaligner.data.datamodel import SpeechSegment
from easyaligner.pipelines import pipeline
from easyaligner.text import load_tokenizer, text_normalizer
from easyaligner.vad.pyannote import load_vad_model

url = "https://www.gutenberg.org/cache/epub/98/pg98.txt"
full_text = requests.get(url).text

# Extract Chapter II text (i.e. the text between "CHAPTER II. The Mail" and "CHAPTER III.")
match = re.search(
    r"(?<=CHAPTER II\.\r\nThe Mail\r\n)[\s\S]+?(?=CHAPTER III\.)",
    full_text,
)
text = match.group().strip()

tutorial_dir = Path("data/tutorials")
filepath_pattern = "tale-of-two-cities_long-en/taleoftwocities_01_dickens_128kb.mp3"

snapshot_download(
    "Lauler/easytranscriber_tutorials",
    repo_type="dataset",
    local_dir=tutorial_dir,
    allow_patterns=filepath_pattern,
)

text = text.strip()

# The alignments will be organized according to how the text is tokenized
tokenizer = load_tokenizer(language="english")  # sentence tokenizer
span_list = list(tokenizer.span_tokenize(text))  # start, end character indices for each sentence

# Chaper II begins 7:06 into the audio, ends 19:54
speeches = [
    [SpeechSegment(speech_id="chapter-ii", text=text, text_spans=span_list, start=426, end=1194)]
]

# Load models and run pipeline
model_vad = load_vad_model()
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h").to("cuda").half()
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# File(s) to align
filepath = Path(tutorial_dir) / filepath_pattern
audio_dir = filepath.parent
audio_files = [filepath.name]

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
