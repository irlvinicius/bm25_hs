# HealthSearch (bm25-hs)

Motor de busca híbrido para documentos clínicos, combinando **BM25** (busca léxica) e **embeddings semânticos**, fundidos via **Reciprocal Rank Fusion (RRF)**.

Desafio Integrador — Tendências em Ciência da Computação (UNIPÊ).

## Funcionalidades

- **Pré-processamento**: normalização, remoção de acentos/pontuação e stopwords em português
- **Busca léxica (BM25)**: parâmetros `k1` e `b` ajustáveis em tempo real
- **Busca semântica**: embeddings multilíngues + similaridade de cosseno
- **Fusão híbrida (RRF)**: combina os dois rankings por posição, com peso `alpha` ajustável
- **Bônus**: re-ranking dos Top-3 com cross-encoder
- Interface interativa em **Streamlit**, com aba de comparação visual dos rankings

## Stack

- Python 3.12
- [uv](https://docs.astral.sh/uv/) — gerenciador de dependências
- Streamlit, pandas, rank_bm25, sentence-transformers

## Como rodar

```bash
uv sync
uv run python -m streamlit run healthsearch_app.py
```

A aplicação abre em `http://localhost:8501`.

## Estrutura

```
bm25_hs/
├── healthsearch_app.py   # aplicação completa (todas as fases)
├── pyproject.toml        # dependências
├── uv.lock                # versões travadas
└── README.md
```

# Documentação

[Relatório técnico completo](docs/relatorio_healthsearch.pdf)