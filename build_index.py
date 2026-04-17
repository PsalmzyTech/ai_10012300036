"""
build_index.py
Run this ONCE to pre-build the FAISS index before starting the Streamlit app.
This avoids the first-run delay inside the app.

Student Name:  Daniel Kingsley Bright Amusah
Index Number:  10012300036
CS4241 - Introduction to Artificial Intelligence

Usage:
    python build_index.py
"""
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)

from rag.pipeline import build_index

if __name__ == "__main__":
    print("Building FAISS index from data/...")
    store = build_index(force_rebuild=False)
    print(f"\nDone. {len(store)} chunks indexed and saved to ./index/")
    print("You can now run:  streamlit run app.py")
