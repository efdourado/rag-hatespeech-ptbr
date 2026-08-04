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

## 2026-08-04 — Fonte e versão do HateBRXplain

- Decisão: usar `dataset/HateBRXplain.csv` do repositório mantido por Francielle
  Vargas, fixado na revisão `0ac05461f17c4d7b7655fbee0391ec325e61a83b`.
- Integridade: SHA-256
  `dd2b669ab8c055805b4813b2628961d951273577fc637039f5eec1238cc9f8bb`.
- Licença adotada: CC BY-NC 4.0, conforme o repositório de origem.
- Motivo: a versão consolidada possui IDs, nomes de coluna estáveis, links das
  publicações para as duas classes e uma declaração explícita de licença.

## 2026-08-04 — Resultado da auditoria inicial

- Resultado: aprovado em todos os controles automatizados.
- Evidências agregadas: 7.000 registros e IDs únicos; 3.500 itens por classe;
  nenhum comentário duplicado exato ou após normalização comparativa; dois
  *rationales* presentes nos 3.500 itens ofensivos e ausentes nos não ofensivos.
- Publicações de origem: 85 valores distintos de `link_post`, dos quais 53
  aparecem nas duas classes.
- Artefato: `outputs/tables/dataset_audit.json`.

## 2026-08-04 — Divisão agrupada e congelada

- Decisão: manter todos os comentários do mesmo `link_post` canônico na mesma
  partição, mesmo que isso impeça uma proporção exatamente igual a 80/10/10.
- Método: `StratifiedGroupKFold` com 10 folds, embaralhamento, semente 42 e
  scikit-learn 1.9.0; fold 0 reservado para teste, fold 7 para validação e os
  demais para treino.
- Resultado: treino com 5.608 itens (2.804 por classe), validação com 706 (353 por
  classe) e teste com 686 (343 por classe).
- Isolamento: 67 publicações no treino, 6 na validação e 12 no teste, sem
  sobreposição de IDs, publicações ou comentários normalizados.
- Manifesto local: `data/processed/split_manifest.csv`, contendo apenas `id`,
  `fold` e `split`; SHA-256
  `6a7212fbb42c26fc8e9c8f1ef3fe391fb9f651d5c14820fc3e2b521002208033`.
- Política RAG: somente IDs marcados como `train` podem ser indexados ou
  recuperados como contexto.
- Motivo: reduzir vazamento temático entre comentários associados à mesma
  publicação, ainda que a divisão deixe de ser diretamente comparável à divisão
  por item usada no artigo original.

## 2026-08-04 — Limitação de concentração por publicação

- Constatação: o maior `link_post` representa 35,6% da validação e 45,9% do teste;
  os três maiores representam, respectivamente, 87,5% e 73,9%.
- Consequência: comentários de uma mesma publicação não serão tratados como
  observações estatisticamente independentes.
- Decisão analítica: intervalos e comparações pareadas devem reamostrar por
  `link_post`; também será reportada uma análise de sensibilidade sem a maior
  publicação do teste.
- Regra: os folds não poderão ser trocados após observar resultados de modelos.

## 2026-08-04 — Piloto de anotação fora do teste

- Decisão: calibrar o guia de sarcasmo com 100 itens da validação, balanceados por
  ofensividade, e congelar o guia antes da anotação do teste.
- Motivo: impedir que decisões de anotação sejam ajustadas após examinar o conjunto
  destinado à avaliação final.
