import os
from transformers import pipeline

# Optional: Load a lightweight Hugging Face model for generation
# Falls back to retrieval-only mode if model not available
MODEL_NAME = os.getenv("FLAN_MODEL", "google/flan-t5-small")

try:
    generator = pipeline("text2text-generation", model=MODEL_NAME)
except Exception as e:
    print(f"[WARN] Could not load model {MODEL_NAME}. Running in retrieval-only mode. Error: {e}")
    generator = None


def build_prompt(question, retrieved):
    """
    Build a context prompt from retrieved text chunks.
    """
    context = "\n\n".join([f"[{r['doc']} p{r['page']}] {r['text']}" for r in retrieved])
    sources = "\n".join(
        [f"[Source {i+1}] {r['doc']} (page {r['page']})" for i, r in enumerate(retrieved)]
    )

    prompt = (
        f"You are StudyMate, an academic assistant.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer clearly, referencing the sources if possible.\n\n"
        f"Sources:\n{sources}"
    )
    return prompt


def generate_answer(question, retrieved, max_tokens=256):
    """
    Generate an answer using Hugging Face pipeline if available,
    otherwise return a retrieval-only response.
    """
    if not retrieved:
        return "No relevant information found in the uploaded documents."

    if generator is None:
        # Retrieval-only mode
        combined = "\n\n".join([f"[{r['doc']} p{r['page']}] {r['text']}" for r in retrieved])
        return f"(Retrieval-only mode)\n\n{combined}"

    prompt = build_prompt(question, retrieved)
    try:
        result = generator(prompt, max_new_tokens=max_tokens, do_sample=False)
        return result[0]["generated_text"]
    except Exception as e:
        return f"(Error during generation: {e})"
