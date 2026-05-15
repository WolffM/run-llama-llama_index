# Bug Reproduction Notes: GoogleGenAIEmbedding aggregated embeddings

## Steps to reproduce

1. Install google-genai SDK version 1.71.0 or later:
   ```
   pip install "google-genai>=1.71.0"
   ```
2. Install the integration package:
   ```
   pip install llama-index-embeddings-google-genai
   ```
3. Run the following code (with a valid `GOOGLE_API_KEY`):
   ```python
   from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

   emb = GoogleGenAIEmbedding(model_name="gemini-embedding-2", api_key="<key>")
   texts = ["Hello world", "This is a test", "Embeddings are useful"]
   embeddings = emb.get_text_embedding_batch(texts)
   print(len(embeddings))  # prints 1 instead of 3
   ```
4. Alternatively, run the unit tests which reproduce the issue without real API
   access:
   ```
   cd llama-index-integrations/embeddings/llama-index-embeddings-google-genai
   pytest tests/test_embeddings_gemini.py::test_batch_embed_calls_embed_content_per_text
   ```

## Observed

With google-genai SDK v1.71.0+, passing a Python `list` of strings to the
`contents` parameter of `client.models.embed_content()` causes the SDK to
aggregate all inputs into a **single** embedding vector and return only one
`EmbeddingResult` object.  As a result, `get_text_embedding_batch(["text1",
"text2", "text3"])` returned a list of length **1** instead of **3**, causing
a `IndexError` or silent data corruption when llama-index tried to pair each
embedding with its source text.

## Expected

Each input text should produce its own independent embedding vector.
`get_text_embedding_batch(["text1", "text2", "text3"])` must return a list of
length **3**, where `result[i]` is the embedding for `texts[i]`.  The fix
calls `embed_content` once per text (passing a single string to `contents`)
so that exactly one embedding is returned per call, then concatenates the
results — preserving the one-to-one mapping between inputs and embeddings
regardless of google-genai SDK version.
