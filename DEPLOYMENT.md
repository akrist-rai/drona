# Deployment Guide

## Option 1: Local Deployment (Your Laptop)

### Prerequisites
```bash
pip install -r requirements_cpu.txt
```

### Run API
```bash
python inference_api.py \
  --model_path newton_mamba_v1.pt \
  --graph_path city_graph.pt \
  --host 0.0.0.0 \
  --port 5000
```

### Test
```bash
curl http://localhost:5000/health
```

---

## Option 2: Render.com (Free Tier)

### Setup

1. **Create a new Web Service** on Render.com
2. **Connect your GitHub repository** (or upload files)
3. **Configure Build & Start Commands**:

   **Build Command:**
   ```bash
   pip install -r requirements_cpu.txt
   ```

   **Start Command:**
   ```bash
   python inference_api.py --model_path newton_mamba_v1.pt --graph_path city_graph.pt --host 0.0.0.0 --port $PORT
   ```

4. **Environment Variables** (optional):
   - `MODEL_PATH`: Path to model file
   - `GRAPH_PATH`: Path to graph file

5. **Upload Model Files**:
   - Upload `newton_mamba_v1.pt` and `city_graph.pt` to your repository
   - Or use Render's persistent disk storage

### Limitations
- **512MB RAM** (free tier)
- **CPU only** (no GPU)
- **Sleeps after 15 minutes** of inactivity (free tier)

---

## Option 3: Railway.app (Free Tier)

### Setup

1. **Create new project** on Railway
2. **Connect GitHub repository**
3. **Add `railway.json`**:
   ```json
   {
     "build": {
       "builder": "NIXPACKS"
     },
     "deploy": {
       "startCommand": "python inference_api.py --model_path newton_mamba_v1.pt --graph_path city_graph.pt --host 0.0.0.0 --port $PORT"
     }
   }
   ```

4. **Set environment variables** in Railway dashboard

---

## Option 4: Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements_cpu.txt .
RUN pip install --no-cache-dir -r requirements_cpu.txt

# Copy application
COPY . .

# Expose port
EXPOSE 5000

# Run API
CMD ["python", "inference_api.py", "--model_path", "newton_mamba_v1.pt", "--graph_path", "city_graph.pt", "--host", "0.0.0.0", "--port", "5000"]
```

### Build & Run
```bash
docker build -t newton-graphmamba-api .
docker run -p 5000:5000 newton-graphmamba-api
```

---

## Production Considerations

### 1. Model Optimization

**Quantization** (reduce memory):
```python
import torch.quantization

model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

**Model Pruning** (reduce size):
```python
# Remove unnecessary parameters
# Use torch.nn.utils.prune
```

### 2. Caching

Add Redis/Memcached for:
- Graph memory caching
- Frequent route predictions

### 3. Load Balancing

Use multiple API instances behind a load balancer:
- Nginx
- Traefik
- Cloudflare

### 4. Monitoring

Add logging and monitoring:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

### 5. Rate Limiting

Add rate limiting to prevent abuse:
```python
from flask_limiter import Limiter
limiter = Limiter(app=app, key_func=get_remote_address)
```

---

## Performance Tips

1. **Batch Inference**: Process multiple requests together
2. **Graph Caching**: Cache graph memory (don't recompute every request)
3. **Model Quantization**: Use INT8 quantization for 4x memory reduction
4. **Async Processing**: Use async Flask (Quart) for better concurrency

---

## Troubleshooting

### Out of Memory
- Reduce model size (`d_model`, `n_layers`)
- Use model quantization
- Process requests one at a time

### Slow Inference
- This is expected on CPU (Mamba is optimized for GPU)
- Consider using a paid GPU instance for production
- Or optimize the `mamba_simple.py` implementation

### Model Not Loading
- Check file paths are correct
- Verify model was saved correctly on Kaggle
- Check PyTorch version compatibility

