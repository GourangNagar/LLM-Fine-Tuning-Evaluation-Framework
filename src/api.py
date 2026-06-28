import time
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

# Configure logging for monitoring hooks
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("InferenceEndpoint")

app = FastAPI(title="LLM Risk Classification API")

# --- Global State for Metrics ---
# In a real app, this would use Prometheus/Grafana or Datadog
metrics = {
    "total_queries": 0,
    "concurrent_queries": 0,
    "average_latency_ms": 0.0,
    "errors": 0
}

# Define request/response models
class QueryRequest(BaseModel):
    narrative: str
    use_ollama: bool = False # Flag to fallback to local Ollama instance for Mac users

class QueryResponse(BaseModel):
    response: str
    latency_ms: float
    model_used: str

# Mock model loading for standard HuggingFace pipeline
# (In production, load model once at startup)
hf_model = None
hf_tokenizer = None

def load_hf_model():
    global hf_model, hf_tokenizer
    if hf_model is None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            
            base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            logger.info("Loading HuggingFace model...")
            hf_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name, 
                device_map="mps" if torch.backends.mps.is_available() else "auto"
            )
            # Try loading LoRA weights
            try:
                hf_model = PeftModel.from_pretrained(base_model, "./finetuned_model_lora")
            except:
                logger.warning("LoRA weights not found. Using base model.")
                hf_model = base_model
        except Exception as e:
            logger.error(f"Failed to load HF model: {e}")

def run_hf_inference(prompt: str) -> str:
    from prompts import get_risk_classification_prompt
    full_prompt = get_risk_classification_prompt(prompt)
    inputs = hf_tokenizer(full_prompt, return_tensors="pt").to(hf_model.device)
    
    import torch
    with torch.no_grad():
        outputs = hf_model.generate(**inputs, max_new_tokens=100, temperature=0.1)
    
    return hf_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

def run_ollama_inference(prompt: str) -> str:
    """Fallback to local Ollama for Mac users running Llama 3 locally."""
    import requests
    from prompts import get_risk_classification_prompt
    
    full_prompt = get_risk_classification_prompt(prompt)
    
    try:
        response = requests.post('http://localhost:11434/api/generate', json={
            "model": "llama3", # Assuming user has 'llama3' pulled in Ollama
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        })
        response.raise_for_status()
        return response.json()['response']
    except Exception as e:
        logger.error(f"Ollama inference failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to connect to local Ollama instance.")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up API and pre-loading models...")
    # Uncomment to load HF model on startup
    # load_hf_model()

@app.post("/generate", response_model=QueryResponse)
async def generate_endpoint(request: QueryRequest):
    start_time = time.time()
    metrics["total_queries"] += 1
    metrics["concurrent_queries"] += 1
    
    try:
        # Simulate handling 50 concurrent queries constraint
        if metrics["concurrent_queries"] > 50:
            logger.warning("High concurrency detected!")
            
        logger.info(f"Processing query. Current concurrent queries: {metrics['concurrent_queries']}")
        
        # Inference Routing
        if request.use_ollama:
            result_text = run_ollama_inference(request.narrative)
            model_used = "Ollama (llama3)"
        else:
            # Fallback to HF if requested (requires load_hf_model to have been run)
            if hf_model is None:
                load_hf_model()
            if hf_model:
                result_text = run_hf_inference(request.narrative)
                model_used = "HuggingFace (TinyLlama+LoRA)"
            else:
                raise HTTPException(status_code=500, detail="HF model not available. Try use_ollama=True.")
        
        # Calculate Latency
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Update metrics (Exponential moving average for latency)
        if metrics["total_queries"] == 1:
            metrics["average_latency_ms"] = latency_ms
        else:
            metrics["average_latency_ms"] = (0.9 * metrics["average_latency_ms"]) + (0.1 * latency_ms)
            
        logger.info(f"Query successful. Latency: {latency_ms:.2f}ms")
        
        return QueryResponse(
            response=result_text,
            latency_ms=latency_ms,
            model_used=model_used
        )
        
    except Exception as e:
        metrics["errors"] += 1
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        metrics["concurrent_queries"] -= 1

@app.get("/metrics")
async def get_metrics():
    """Endpoint for tracking latency and accuracy drift proxies."""
    return metrics

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
