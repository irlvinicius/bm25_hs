"""
HealthSearch — Motor de Busca Híbrido (BM25 + Semântico + RRF)
Desafio Integrador — UNIPÊ

Fase 1: Ingestão e Pré-processamento
Fase 2: Motor de Busca Léxico (BM25)
Fase 3: Motor de Busca Semântico (Embeddings + Cosseno)
Fase 4: Fusão Híbrida (Reciprocal Rank Fusion)
Bônus:  Re-Ranking com Cross-Encoder nos Top-3 do RRF
"""

import re
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------------------------------------------------------------------------
# FASE 1.1 — CORPUS CLÍNICO (hardcoded, 6 documentos fixos)
# ---------------------------------------------------------------------------
CORPUS = [
    {
        "id": 1,
        "titulo": "Eletrocardiograma (ECG) — Interpretação Básica",
        "texto": (
            "O eletrocardiograma registra a atividade elétrica do coração através de "
            "eletrodos posicionados na pele. A onda P representa a despolarização "
            "atrial, o complexo QRS representa a despolarização ventricular e a onda T "
            "representa a repolarização ventricular. Alterações no segmento ST podem "
            "indicar isquemia ou infarto agudo do miocárdio. O intervalo PR prolongado "
            "sugere bloqueio atrioventricular."
        ),
    },
    {
        "id": 2,
        "titulo": "Farmacologia Cardíaca — Betabloqueadores e IECA",
        "texto": (
            "Os betabloqueadores reduzem a frequência cardíaca e a contratilidade "
            "miocárdica, sendo indicados após infarto e em insuficiência cardíaca. Os "
            "inibidores da enzima conversora de angiotensina (IECA) diminuem a pós-carga "
            "e são primeira linha no tratamento da hipertensão arterial e da "
            "insuficiência cardíaca. Efeitos adversos incluem tosse seca e hipotensão."
        ),
    },
    {
        "id": 3,
        "titulo": "Hipertensão Arterial Sistêmica — Manejo Clínico",
        "texto": (
            "A hipertensão arterial é definida por pressão sistólica igual ou superior "
            "a 140 mmHg ou diastólica igual ou superior a 90 mmHg. É um dos principais "
            "fatores de risco para acidente vascular cerebral e infarto do miocárdio. O "
            "tratamento envolve mudanças no estilo de vida e, quando necessário, "
            "medicações como diuréticos, IECA e bloqueadores de canal de cálcio."
        ),
    },
    {
        "id": 4,
        "titulo": "Acidente Vascular Cerebral (AVC) — Reconhecimento e Conduta",
        "texto": (
            "O acidente vascular cerebral isquêmico ocorre pela obstrução de um vaso "
            "sanguíneo cerebral, enquanto o hemorrágico decorre de ruptura vascular. Os "
            "sinais clássicos incluem desvio de rima labial, fraqueza em um dos lados do "
            "corpo e alteração na fala. O tempo até o atendimento é crítico para reduzir "
            "sequelas neurológicas — a janela para trombólise costuma ser de até 4,5 horas."
        ),
    },
    {
        "id": 5,
        "titulo": "Ressuscitação Cardiopulmonar (RCP) — Protocolo de Suporte Básico",
        "texto": (
            "A ressuscitação cardiopulmonar deve ser iniciada imediatamente após a "
            "identificação de parada cardiorrespiratória, com compressões torácicas de "
            "alta qualidade a uma frequência de 100 a 120 por minuto. O uso precoce do "
            "desfibrilador externo automático aumenta significativamente a sobrevida em "
            "casos de fibrilação ventricular. A síndrome coronariana aguda é uma das "
            "principais causas de parada súbita."
        ),
    },
    {
        "id": 6,
        "titulo": "Monitorização em Unidade de Terapia Intensiva (UTI)",
        "texto": (
            "Pacientes críticos internados em UTI requerem monitorização contínua de "
            "frequência cardíaca, pressão arterial invasiva, saturação de oxigênio e "
            "débito urinário. Escalas como o APACHE II auxiliam na estimativa de "
            "gravidade e mortalidade. A vigilância de arritmias e de sinais precoces de "
            "choque é essencial para intervenção rápida na unidade de terapia intensiva."
        ),
    },
]

# ---------------------------------------------------------------------------
# FASE 1.2 — STOPWORDS EM PORTUGUÊS (lista enxuta, hardcoded)
# ---------------------------------------------------------------------------
STOPWORDS_PT = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "por", "para", "com", "sem", "sob", "sobre", "entre",
    "e", "ou", "mas", "que", "se", "ao", "aos", "à", "às",
    "é", "foi", "ser", "está", "são", "como", "mais", "menos",
}

# Modelos usados na Fase 3 e no bônus
MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"
MODELO_CROSS_ENCODER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
K_RRF = 60  # constante de suavização do Reciprocal Rank Fusion


# ---------------------------------------------------------------------------
# FASE 1.3 — PIPELINE DE PRÉ-PROCESSAMENTO
# ---------------------------------------------------------------------------
def normalizar_acentos(texto: str) -> str:
    """Remove acentuação (ó -> o) preservando o restante do texto."""
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def preprocessar(texto: str) -> list[str]:
    """
    Pipeline: minúsculas -> remove acentos -> remove pontuação ->
    tokeniza -> remove stopwords.
    Retorna a lista de tokens prontos para o BM25.
    """
    texto = texto.lower()
    texto = normalizar_acentos(texto)
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    tokens = texto.split()
    tokens = [t for t in tokens if t not in STOPWORDS_PT]
    return tokens


@st.cache_data
def carregar_corpus_processado():
    """Pré-processa o corpus inteiro uma única vez (cacheado pelo Streamlit)."""
    corpus_processado = []
    for doc in CORPUS:
        tokens = preprocessar(doc["texto"])
        corpus_processado.append({**doc, "tokens": tokens})
    return corpus_processado


# ---------------------------------------------------------------------------
# FASE 2 — MOTOR DE BUSCA BM25
# ---------------------------------------------------------------------------
def buscar_bm25(query: str, corpus: list[dict], k1: float, b: float) -> list[dict]:
    """
    Executa a busca BM25 sobre o corpus pré-processado.
    Retorna os documentos ordenados por score (maior -> menor),
    cada um com 'score_bm25' e 'rank_bm25' adicionados.
    """
    tokens_corpus = [doc["tokens"] for doc in corpus]
    bm25 = BM25Okapi(tokens_corpus, k1=k1, b=b)

    query_tokens = preprocessar(query)
    scores = bm25.get_scores(query_tokens)

    resultados = [{**doc, "score_bm25": float(score)} for doc, score in zip(corpus, scores)]
    resultados.sort(key=lambda d: d["score_bm25"], reverse=True)
    for i, doc in enumerate(resultados, start=1):
        doc["rank_bm25"] = i

    return resultados


# ---------------------------------------------------------------------------
# FASE 3 — MOTOR DE BUSCA SEMÂNTICO
# ---------------------------------------------------------------------------
@st.cache_resource
def carregar_modelo_embeddings():
    """Carrega o modelo de embeddings uma única vez por sessão (é pesado)."""
    return SentenceTransformer(MODELO_EMBEDDINGS)


@st.cache_data
def gerar_embeddings_corpus(_modelo, textos: tuple[str, ...]) -> np.ndarray:
    """
    Gera os embeddings do corpus uma única vez.
    O underscore em '_modelo' diz ao Streamlit para NÃO tentar hashear o
    modelo (objeto pesado/não hasheável) — o cache é baseado só em 'textos'.
    """
    return _modelo.encode(list(textos), normalize_embeddings=True)


def similaridade_cosseno(query_emb: np.ndarray, corpus_emb: np.ndarray) -> np.ndarray:
    """
    Como os embeddings já vêm normalizados (normalize_embeddings=True),
    o produto escalar É a similaridade de cosseno.
    """
    return corpus_emb @ query_emb


def buscar_semantico(query: str, corpus: list[dict], modelo) -> list[dict]:
    """
    Executa a busca semântica via embeddings + similaridade de cosseno.
    Retorna os documentos ordenados por score (maior -> menor),
    cada um com 'score_semantico' e 'rank_semantico' adicionados.
    """
    textos = tuple(doc["texto"] for doc in corpus)
    corpus_emb = gerar_embeddings_corpus(modelo, textos)
    query_emb = modelo.encode(query, normalize_embeddings=True)

    scores = similaridade_cosseno(query_emb, corpus_emb)

    resultados = [{**doc, "score_semantico": float(score)} for doc, score in zip(corpus, scores)]
    resultados.sort(key=lambda d: d["score_semantico"], reverse=True)
    for i, doc in enumerate(resultados, start=1):
        doc["rank_semantico"] = i

    return resultados


# ---------------------------------------------------------------------------
# FASE 4 — FUSÃO HÍBRIDA (RECIPROCAL RANK FUSION)
# ---------------------------------------------------------------------------
def fundir_rrf(
    resultados_bm25: list[dict],
    resultados_semanticos: list[dict],
    alpha: float,
    k_rrf: int = K_RRF,
) -> list[dict]:
    """
    Combina os dois rankings usando a fórmula:

        score_rrf(D) = alpha * 1/(k_rrf + rank_bm25(D))
                     + (1-alpha) * 1/(k_rrf + rank_semantico(D))

    Note que a fusão usa apenas a POSIÇÃO (rank) em cada lista, nunca o
    score bruto — isso evita comparar escalas incompatíveis (score do BM25
    é ilimitado, score do cosseno vai de -1 a 1).
    """
    ranks_bm25 = {doc["id"]: doc["rank_bm25"] for doc in resultados_bm25}
    ranks_semantico = {doc["id"]: doc["rank_semantico"] for doc in resultados_semanticos}
    docs_por_id = {doc["id"]: doc for doc in resultados_bm25}

    fundidos = []
    for doc_id, doc in docs_por_id.items():
        r_bm25 = ranks_bm25[doc_id]
        r_sem = ranks_semantico[doc_id]
        score_rrf = alpha * (1 / (k_rrf + r_bm25)) + (1 - alpha) * (1 / (k_rrf + r_sem))
        fundidos.append({
            **doc,
            "rank_bm25": r_bm25,
            "rank_semantico": r_sem,
            "score_rrf": score_rrf,
        })

    fundidos.sort(key=lambda d: d["score_rrf"], reverse=True)
    for i, doc in enumerate(fundidos, start=1):
        doc["rank_final"] = i

    return fundidos


# ---------------------------------------------------------------------------
# BÔNUS — RE-RANKING COM CROSS-ENCODER (TOP-3 DO RRF)
# ---------------------------------------------------------------------------
@st.cache_resource
def carregar_cross_encoder():
    return CrossEncoder(MODELO_CROSS_ENCODER)


def reranquear_top3(query: str, resultados_rrf: list[dict], modelo_ce) -> list[dict]:
    """
    Pega o Top-3 do RRF e reordena usando um cross-encoder, que avalia o
    par (query, documento) diretamente — mais preciso que embeddings, mas
    caro demais pra rodar no corpus inteiro. Por isso só nos 3 finalistas.
    """
    top3 = resultados_rrf[:3]
    resto = resultados_rrf[3:]

    pares = [(query, doc["texto"]) for doc in top3]
    scores_ce = modelo_ce.predict(pares)

    for doc, score in zip(top3, scores_ce):
        doc["score_cross_encoder"] = float(score)

    top3_reranqueado = sorted(top3, key=lambda d: d["score_cross_encoder"], reverse=True)
    return top3_reranqueado + resto


# ---------------------------------------------------------------------------
# INTERFACE
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="HealthSearch", layout="wide")
    st.title("HealthSearch — Motor de Busca Híbrido")
    st.caption("BM25 (léxico) + Embeddings (semântico) fundidos via Reciprocal Rank Fusion")

    corpus = carregar_corpus_processado()

    with st.sidebar:
        st.header("Parâmetros do BM25")
        k1 = st.slider(
            "k1 (saturação de frequência)", 0.0, 3.0, 1.5, 0.1,
            help="k1=0 ignora repetição do termo; valores altos dão mais peso a termos repetidos.",
        )
        b = st.slider(
            "b (normalização por tamanho)", 0.0, 1.0, 0.75, 0.05,
            help="b=0 desliga a normalização por tamanho do documento; b=1 normaliza totalmente.",
        )

        st.header("Fusão híbrida (RRF)")
        alpha = st.slider(
            "alpha (peso do BM25 vs. semântico)", 0.0, 1.0, 0.5, 0.05,
            help="alpha=1 usa só o BM25; alpha=0 usa só o semântico.",
        )

        st.header("Bônus")
        usar_cross_encoder = st.checkbox(
            "Re-ranking com Cross-Encoder no Top-3",
            help="Reordena os 3 melhores resultados do RRF com um modelo mais preciso (e mais lento).",
        )

    query = st.text_input("Digite sua busca:", placeholder="ex: infarto do miocárdio")

    if not query:
        st.info("Digite uma busca acima para começar.")
        return

    modelo_embeddings = carregar_modelo_embeddings()

    resultados_bm25 = buscar_bm25(query, corpus, k1, b)
    resultados_semanticos = buscar_semantico(query, corpus, modelo_embeddings)
    resultados_rrf = fundir_rrf(resultados_bm25, resultados_semanticos, alpha)

    if usar_cross_encoder:
        modelo_ce = carregar_cross_encoder()
        resultados_rrf = reranquear_top3(query, resultados_rrf, modelo_ce)

    aba_resultados, aba_comparacao = st.tabs(["Resultados", "Comparação de rankings"])

    with aba_resultados:
        for doc in resultados_rrf:
            with st.container(border=True):
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.markdown(f"**#{doc['rank_final']} — {doc['titulo']}**")
                    st.caption(doc["texto"][:220] + "...")
                    legenda = f"BM25: #{doc['rank_bm25']} · Semântico: #{doc['rank_semantico']}"
                    if "score_cross_encoder" in doc:
                        legenda += f" · Cross-encoder: {doc['score_cross_encoder']:.3f}"
                    st.caption(legenda)
                with col_b:
                    st.metric("Score RRF", f"{doc['score_rrf']:.4f}")

    with aba_comparacao:
        st.markdown("Posição (rank) de cada documento em cada método de busca:")
        df = pd.DataFrame([
            {
                "Documento": doc["titulo"][:30],
                "BM25": doc["rank_bm25"],
                "Semântico": doc["rank_semantico"],
                "RRF (final)": doc["rank_final"],
            }
            for doc in resultados_rrf
        ]).set_index("Documento")
        st.bar_chart(df)
        st.caption("Quanto menor a barra, melhor o ranking (posição 1 = melhor).")


if __name__ == "__main__":
    main()