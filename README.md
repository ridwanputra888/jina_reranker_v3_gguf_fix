# fixing Jina Reranker v3 GGUF Implementation

fixing https://huggingface.co/jinaai/jina-reranker-v3-GGUF missing binary embedder and tokenizer

## Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) package manager
- C++ compiler (for building llama.cpp)
- CMake

## Quick Setup

### 1. Install Dependencies

```bash
uv sync
```



2. Build llama.cpp Binaries
```bash
# Clone and build llama.cpp with embedding support
git clone https://github.com/hanxiao/llama.cpp
mkdir llama.cpp/build
cmake -S llama.cpp -B llama.cpp/build -DLLAMA_BUILD_SERVER=ON -DLLAMA_EMBEDDINGS=ON -DLLAMA_BUILD_EXAMPLES=ON
make -C llama.cpp/build -j$(nproc)

# Verify the binaries are built
ls -la llama.cpp/build/bin/llama-embedding
ls -la llama.cpp/build/bin/llama-tokenize

# Make sure they're executable
chmod +x llama.cpp/build/bin/llama-embedding
chmod +x llama.cpp/build/bin/llama-tokenize

# Copy binaries to current directory (optional)
cp llama.cpp/build/bin/llama-embedding llama-embedding
cp llama.cpp/build/bin/llama-tokenize llama-tokenize
```

### 2. Download Model and tensor file
```bash
wget https://huggingface.co/jinaai/jina-reranker-v3-GGUF/resolve/main/jina-reranker-v3-Q8_0.gguf

wget https://huggingface.co/jinaai/jina-reranker-v3-GGUF/resolve/main/projector.safetensors
```

### 3. modify rerank.py 
```python 
GGUFReranker(
    model_path: str = "jina-reranker-v3-BF16.gguf",
    projector_path: str = "projector.safetensors", 
    llama_embedding_path: str = ".//llama-embedding",
    llama_tokenize_path: str = "./llama-tokenize"
)

rerank(
    query: str,
    documents: List[str],
    top_n: Optional[int] = None,
    return_embeddings: bool = False,
    instruction: Optional[str] = None
) -> List[Dict]
```

### 4. implementation
```python
from rerank import GGUFReranker

# Initialize reranker
reranker = GGUFReranker(
    model_path="jina-reranker-v3-Q8_0.gguf",
    projector_path="projector.safetensors",
    llama_embedding_path="./llama-embedding",
    llama_tokenize_path="./llama-tokenize"
)

# Rerank documents
query = "Apa ibu kota Perancis?"
documents = [
    "Paris adalah ibu kota dan kota terbesar di Perancis.",
    "Berlin adalah ibu kota Jerman.",
    "Menara Eiffel terletak di Paris."
]

results = reranker.rerank(query, documents)

for result in results:
    print(f"Score: {result['relevance_score']:.4f}, Doc: {result['document'][:50]}...")
```
```bash
Score: 0.3942, Doc: Paris adalah ibu kota dan kota terbesar di Peranci...
Score: -0.0864, Doc: Menara Eiffel terletak di Paris....
Score: -0.1263, Doc: Berlin adalah ibu kota Jerman....
```