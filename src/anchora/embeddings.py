"""Embeddings: a local Ollama model, with a deterministic hashing fallback.

The ``hash`` provider needs no model or network, so tests and CI are fully
reproducible and the project runs offline end-to-end.
"""

from __future__ import annotations

import hashlib
import math

import httpx

from anchora.config import settings


def embed_texts(
    texts: list[str], provider: str | None = None, *, query: bool = False
) -> list[list[float]]:
    """Embed each text into a unit-norm vector.

    ``query=True`` enables the cross-lingual bridge of the ``hash`` provider so
    that English questions can retrieve the Portuguese corpus offline (see
    ``_GLOSSARY``). It is a no-op for the Ollama provider, which is multilingual
    natively, and for the corpus side (which is never bridged).
    """
    chosen = provider or settings.embed_provider
    if chosen == "hash":
        return [_hash_embed(text, query=query) for text in texts]
    return [_ollama_embed(text) for text in texts]


def _ollama_embed(text: str) -> list[float]:
    response = httpx.post(
        f"{settings.ollama_base_url}/api/embeddings",
        json={"model": settings.embed_model, "prompt": text},
        timeout=settings.request_timeout,
    )
    response.raise_for_status()
    vector = [float(value) for value in response.json()["embedding"]]
    return _normalize(vector)


def _hash_embed(text: str, *, query: bool = False) -> list[float]:
    """Signed hashing trick → deterministic bag-of-words embedding.

    Tokens are lowercased and accent-folded so that, e.g., ``prazo`` and
    ``Prazo`` collide — useful for short legal queries.
    """
    dim = settings.embed_dim
    vector = [0.0] * dim
    for token in tokenize(text, query=query):
        digest = int(hashlib.sha256(token.encode()).hexdigest(), 16)
        vector[digest % dim] += 1.0 if (digest >> 8) % 2 == 0 else -1.0
    return _normalize(vector)


# Accent-folded stopwords (Portuguese + English): dropped so distinctive legal
# terms dominate the bag-of-words signal instead of high-frequency glue words.
# English words are included because questions are in English while the corpus
# is in Portuguese (see ``_GLOSSARY``).
# fmt: off
_STOPWORDS = frozenset({
    # Portuguese
    "a", "ao", "aos", "as", "ate", "com", "como", "da", "das", "de", "do",
    "dos", "e", "em", "entre", "era", "essa", "esse", "esta", "este", "eu",
    "foi", "foram", "ha", "isso", "ja", "la", "lhe", "mais", "mas", "me",
    "mesmo", "muito", "na", "nao", "nas", "no", "nos", "num", "numa", "o",
    "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "qual",
    "quais", "quando", "que", "quem", "se", "sem", "ser", "sao", "sobre",
    "sua", "suas", "seu", "seus", "tem", "um", "uma", "uns", "voce",
    # English
    "an", "and", "after", "are", "be", "by", "can", "concluded", "did",
    "does", "for", "from", "has", "have", "how", "in", "into", "is",
    "its", "long", "many", "must", "new", "of", "on", "once", "one", "or",
    "over", "per", "take", "that", "the", "their", "there", "this", "to",
    "under", "was", "were", "what", "which", "who", "whose", "with", "years",
})
# fmt: on

# Deterministic English -> Portuguese bridge for the offline ``hash`` provider.
# The corpus is real Brazilian law (Portuguese); questions are English. This
# lexical map lets English query terms collide with the corpus' distinctive
# legal vocabulary so retrieval recall holds in CI without any model or network.
# The production path (Ollama ``nomic-embed-text``) is multilingual and needs
# none of this. Values are accent-folded to match corpus tokenization.
# fmt: off
_GLOSSARY: dict[str, tuple[str, ...]] = {
    # General legal vocabulary
    "deadline": ("prazo",), "procedure": ("processo",),
    # Access to information (LAI)
    "access": ("acesso",), "information": ("informacao",), "request": ("pedido",),
    "respond": ("responder",), "answer": ("responder",), "agency": ("orgao",),
    "secrecy": ("sigilo",), "reserved": ("reservada",), "secret": ("secreta",),
    "topsecret": ("ultrassecreta",), "denied": ("negado",), "appeal": ("recurso",),
    "authority": ("autoridade",), "higher": ("superior",),
    # Civil service (Lei 8.112)
    "servant": ("servidor",), "appointment": ("nomeacao",), "office": ("cargo",),
    "published": ("publicacao",), "probationary": ("estagio", "probatorio"),
    "disciplinary": ("disciplinares",), "penalties": ("penalidades",),
    # Public Defender's Office (LC 80)
    "defender": ("defensoria", "defensor"), "principles": ("principios",),
    "institutional": ("institucionais",), "aid": ("assistencia",),
    "assistance": ("assistencia",), "career": ("carreira",), "enter": ("ingresso",),
    # Data protection (LGPD)
    "sensitive": ("sensivel",), "data": ("dados",), "processing": ("tratamento",),
    "agents": ("agentes",), "subject": ("titular",), "rights": ("direitos",),
    # Procedural deadlines (CPC)
    "procedural": ("processuais",), "business": ("uteis",), "calendar": ("corridos",),
    "contestation": ("contestacao",), "double": ("dobro",), "filings": ("manifestacoes",),
    # Procurement (Lei 14.133)
    "bidding": ("licitacao",), "procurement": ("licitacao", "contratacoes"),
    "modalities": ("modalidades",), "abolished": ("extintas",),
    "centralized": ("centralizada",), "disclosure": ("divulgacao",),
    # Administrative process (Lei 9.784)
    "administrative": ("administrativo",), "administration": ("administracao",),
    "decide": ("decidir",), "instruction": ("instrucao",), "annul": ("anular",),
    "acts": ("atos",), "favorable": ("favoraveis",), "lapse": ("decai",),
    # Free legal aid (gratuidade da justiça)
    "free": ("gratuidade",), "justice": ("justica",), "costs": ("custas",),
    "fees": ("honorarios",), "expenses": ("despesas",), "beneficiary": ("beneficiario",),
    "enforceability": ("exigibilidade",), "suspended": ("suspensa", "suspensiva"),
    "person": ("pessoa",),
}
# fmt: on


def bridge_tokens(tokens: list[str]) -> list[str]:
    """Expand English legal terms to their Portuguese corpus equivalents."""
    bridged: list[str] = []
    for token in tokens:
        bridged.extend(_GLOSSARY.get(token, (token,)))
    return bridged


def tokenize(text: str, *, query: bool = False) -> list[str]:
    """Accent-folded content tokens (duplicates kept — BM25 needs frequencies).

    Shared by the ``hash`` embedding provider and the BM25 index in
    :mod:`anchora.lexical`, so dense-offline and lexical retrieval see the
    exact same token stream (including the English→Portuguese query bridge).
    """
    folded = _strip_accents(text.lower())
    tokens = [token for token in _split_words(folded) if token and token not in _STOPWORDS]
    return bridge_tokens(tokens) if query else tokens


def _split_words(text: str) -> list[str]:
    return ["".join(ch for ch in word if ch.isalnum()) for word in text.split()]


def _strip_accents(text: str) -> str:
    import unicodedata

    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]
