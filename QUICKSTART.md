# Quick Start Guide

## 🚀 5-Minute Setup

### Step 1: Train on Kaggle (5 minutes)

1. **Go to [Kaggle.com](https://www.kaggle.com)** and create a new Notebook
2. **Enable GPU**: Settings → Accelerator → GPU T4 x2
3. **Upload files** to your Kaggle notebook:
   - `newton_graphmamba.py`
   - `graph_utils.py`
   - Copy cells from `kaggle_training.ipynb`
4. **Run the first cell** (Magic Install Block) - this takes ~5 minutes
5. **Run remaining cells** sequentially
6. **Download** `newton_mamba_v1.pt` from Output section

### Step 2: Run Inference Locally (2 minutes)

```bash
# Install dependencies
pip install -r requirements_cpu.txt

# Download your model file (from Kaggle)
# Place it in the project directory

# Run API
python inference_api.py --model_path newton_mamba_v1.pt
```

### Step 3: Test It

```bash
# Health check
curl http://localhost:5000/health

# Predict next node
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"route": [1, 2, 3], "vehicle_id": 0, "top_k": 5}'
```

## 📁 File Checklist

Before training on Kaggle, make sure you have:

- [x] `newton_graphmamba.py` - Model implementation
- [x] `graph_utils.py` - Graph conversion utilities
- [x] `kaggle_training.ipynb` - Training notebook

For inference:

- [x] `mamba_simple.py` - CPU-compatible Mamba
- [x] `inference_api.py` - Flask API
- [x] `requirements_cpu.txt` - CPU dependencies
- [x] `newton_mamba_v1.pt` - Trained model (from Kaggle)

## 🐛 Common Issues

### "ModuleNotFoundError: No module named 'mamba_ssm'"
- **Solution**: Run the Magic Install Block cell first on Kaggle

### "CUDA out of memory"
- **Solution**: Reduce `batch_size` or `d_model` in training notebook

### "Model file not found"
- **Solution**: Make sure you downloaded `newton_mamba_v1.pt` from Kaggle Output

### "API is slow"
- **Expected**: CPU inference is slower than GPU. This is normal.
- **Solution**: Use a GPU instance for production, or optimize `mamba_simple.py`

## 📚 Next Steps

1. **Replace synthetic data** with real trajectory data in training notebook
2. **Tune hyperparameters** (`d_model`, `n_layers`, etc.)
3. **Deploy to production** (see `DEPLOYMENT.md`)
4. **Add monitoring** and logging

## 💡 Tips

- **Kaggle**: You get 30 hours/week of free GPU time
- **Model Size**: Keep `d_model` ≤ 256 for CPU inference
- **Graph Size**: Large graphs (>100k nodes) may need optimization
- **Batch Size**: Start with 32, reduce if OOM errors occur

## 🎯 Expected Results

After training:
- Model file: `newton_mamba_v1.pt` (~50-200 MB)
- Training loss should decrease over epochs
- Inference API responds in <1 second per request (CPU)

---

**Need help?** Check `README.md` for detailed documentation.

