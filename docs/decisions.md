# Registro de decisões

## 2026-07-20 — Nome do projeto

- Decisão: usar `rag-hatespeech-ptbr`.
- Motivo: nome escolhido pelo autor para o repositório em `github.com/efdourado`.

## 2026-07-20 — Separação entre ofensividade e sarcasmo

- Status: consolidada na versão final do PCC de 22 de julho de 2026.
- Decisão: tratar ofensividade como alvo e sarcasmo como estrato de avaliação.
- Motivo: HateBRXplain fornece rótulos e *rationales* de ofensividade, não rótulos
  verdadeiros de sarcasmo.

## 2026-07-20 — Dependências de RAG

- Decisão: adiar LangChain, Pinecone e clientes de LLM.
- Motivo: primeiro será auditado o dataset e congelado o protocolo.

## 2026-07-22 — Ordem de implementação no TCC

- Decisão: concluir aquisição, contrato, auditoria, identificação e divisão dos
  dados antes de implementar embeddings, Pinecone, LangChain, FastAPI ou LLM.
- Motivo: a divisão define quais exemplos podem entrar no índice RAG e evita
  vazamento experimental; escolhas de infraestrutura não devem anteceder a
  validação do corpus.

## 2026-07-22 — Imutabilidade dos dados brutos

- Decisão: preservar o arquivo oficial em `data/raw/` sem alterações e produzir
  quaisquer correções somente em `data/interim/` ou `data/processed/`.
- Motivo: garantir rastreabilidade, reprodutibilidade e comparação com a fonte.
