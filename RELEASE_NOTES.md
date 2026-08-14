# FOG Latent Reasoner v1.0.0-alpha: 10M Pretrained Release

This is the first official release of **FOG Latent Reasoner** (Finite Operator Grammar Latent Memory & Workspaces), featuring a 10-million parameter model fully pretrained on the TinyStories dataset for 10,000 steps (~34 million tokens).

## 🚀 Key Highlights & Results
- **Architecture**: 10M parameter causal language model leveraging FOG lexical architecture and latent recurrent workspaces.
- **Training Duration**: 10,000 steps with linear warmup and cosine decay.
- **Convergence Metrics**:
  - **Final Training Loss**: 2.29
  - **Final Validation Loss**: 2.62
  - **Perplexity (PPL)**: 9.91
  - **Token Accuracy**: 49.88%
- **Computational Efficiency**: Trained on CPU with a sustained throughput of ~3,000 tokens/sec.

## 📦 Assets
- `fog_10m_pretrain_10k.pt`: Pretrained model weights (99 MB) trained for 10k steps on TinyStories.

## 🛠️ Quick Start
To run inference or test generation using this checkpoint:
```bash
python3 test_gen.py --checkpoint checkpoints/pretrain_large/best.pt
```
