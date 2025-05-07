# LangChain Groq App

## Overview

This repo contains two scripts:

- **preprocess.py** – builds & saves a FAISS index from your documents.
- **app.py** – a Streamlit app that loads the index and runs a Groq‐powered QA interface.

## Prerequisites

- A virtual environment (recommended)
- Create a `.env` file in the same folder, set:
  ```
  GROQ_API_KEY=your_api_key_here
  ```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) review `preprocess.py` & `app.py` to adjust chunk sizes or source URLs.

## Usage

1. Build and persist the FAISS index:

   ```bash
   python preprocess.py
   ```

   This creates `faiss_store/index.faiss` in the repo folder.

2. Launch the Streamlit app:
   ```bash
   python -m streamlit run app.py
   ```
   - The app will load the prebuilt index on startup.
   - Enter a question in the input box and hit Enter.

## Rebuilding the Index

If you change source documents or embedding settings, re-run:

```bash
python preprocess.py
```

## Security

We use `allow_dangerous_deserialization=True` in `app.py`. Only run this if you trust that `faiss_store/` was generated locally.
