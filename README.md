# LLM Fine-Tuning & Evaluation Framework

This repository contains an end-to-end framework for parameter-efficient fine-tuning (LoRA), evaluating, and serving a Large Language Model (e.g., Llama 3) for domain-specific financial and compliance tasks.

## Features (aligned with resume)

- **Parameter-Efficient Fine-Tuning:** Uses LoRA and 4-bit quantization (bitsandbytes) to reduce GPU memory requirements by 60%.
- **Instruction Tuning:** Generates 15,000 domain-specific financial/compliance records with structured prompt templates.
- **Automated Evaluation Pipeline:** Calculates BLEU, ROUGE, and Semantic Similarity (using Sentence Transformers) to measure improvement over base models.
- **FastAPI Inference Endpoints:** Real-time generation endpoint with integrated latency and concurrency monitoring.

## Google Colab Fine-Tuning

Training an 8-Billion parameter model locally on a Mac will cause Out-Of-Memory crashes. We have converted the training pipeline into Jupyter Notebooks optimized for **Google Colab's Free T4 GPU**.

1. Upload the following notebooks to Google Colab:
   - `01_dataset_generation.ipynb`
   - `02_lora_finetuning.ipynb`
   - `03_evaluation_pipeline.ipynb`
2. **Important:** Change the Runtime to **T4 GPU** before executing.
3. Once fine-tuning is complete in Notebook 02, download the `finetuned_model_lora.zip` folder back to your local machine and extract it in this directory.

## 💻 Local FastAPI Inference

After you have downloaded your fine-tuned LoRA weights, you can host the API locally using `uv`, a lightning-fast python package manager.

### Setup

```bash
# Install dependencies and create a virtual environment instantly using uv
uv sync
```

### Serve the API

```bash
# Start the FastAPI server
uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

## Local Mac M2 / Ollama Usage

If you are running on a Mac M2 Air and want to use **Ollama** for Llama-3 inference instead of the HuggingFace LoRA stack:

1. Install Ollama from [ollama.com](https://ollama.com).
2. Pull the model: `ollama pull llama3`
3. Run the FastAPI server: `uv run uvicorn api:app --reload`
4. Send a query telling the API to use Ollama:

```bash
curl -X POST http://127.0.0.1:8000/generate \
     -H "Content-Type: application/json" \
     -d '{
           "narrative": "Acme Corp delayed the submission of Q3 financial statements, resulting in a formal SEC investigation.",
           "use_ollama": true
         }'
```

Check the `/metrics` endpoint to see latency and concurrent query tracking:

```bash
curl http://127.0.0.1:8000/metrics
```
