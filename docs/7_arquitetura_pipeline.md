# Arquitetura do pipeline experimental

Este documento descreve, componente a componente, o pipeline necessário
para executar o experimento descrito no PCC/protocolo de pesquisa. Cada
componente tem um estado (feito, planejado, bloqueado) e uma justificativa
de design. O critério geral: preferir a solução mais simples que resolva o
problema real, e só adicionar uma dependência nova quando uma justificativa
específica exigir.

## Visão geral do fluxo

```
data/raw/HateBRXplain.csv
        │
        ▼
  inspect_dataset.py  ──────────────► outputs/tables/dataset_audit.json
        │
        ▼
  create_splits.py  ────────────────► data/processed/split_manifest.csv
        │                              outputs/tables/split_audit.json
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │ split_manifest.csv define: train / validation / test     │
  └─────────────────────────────────────────────────────────┘
        │
        ├──► run_baseline.py (TF-IDF + LogReg, treino → validação)
        │
        ├──► [piloto de sarcasmo: 100 itens da validação, anotação humana]
        │
        ├──► [índice RAG: apenas split == "train"]
        │
        ├──► [baseline Gemini sem recuperação]
        │
        ├──► [RAG: recuperação + prompt + geração via Gemini]
        │
        └──► [avaliação final única no teste + estrato sarcástico]
```

## Componentes

### 1. Ingestão — feito

`data/raw/HateBRXplain.csv`, obtido manualmente pelo autor (instruções no
README), hash fixado em 12_decisoes.md. Não há script de download
automatizado por decisão deliberada: o dataset tem licença CC BY-NC 4.0 e
não deve ser redistribuído nem baixado silenciosamente sem que o usuário
veja de onde vem.

### 2. Auditoria do dataset — feito

`scripts/inspect_dataset.py` (autocontido). Produz
`outputs/tables/dataset_audit.json`, um relatório agregado (sem texto de
comentários) com verificações binárias (`checks`). Decisão de design: o
relatório é sempre determinístico e seguro para versionar, porque nunca
inclui texto bruto — apenas contagens.

### 3. Normalização mínima — feito, por decisão deliberada de escopo mínimo

`normalize_for_comparison()` (duplicada intencionalmente em
`scripts/inspect_dataset.py` e `src/rag_hatespeech_ptbr/splits.py`, ver
8_auditoria_vazamento.md para a razão) normaliza Unicode (NFKC), *casefold*
e colapsa espaços — apenas para comparação, nunca sobrescreve o texto
original. Isso é suficiente para detectar duplicatas quase-idênticas sem
correr o risco de alterar sinais linguísticos (caixa, pontuação, emojis)
que a Etapa 1 do plano de dados explicitamente proíbe remover.

### 4. IDs estáveis — feito

O próprio `id` do HateBRXplain já é estável, único e não textual — não foi
necessário gerar um novo. Verificado em `dataset_audit.json`
(`ids_are_complete_and_unique: true`).

### 5. Deduplicação — feito (detecção); decisão de tratamento não foi necessária

Zero duplicatas exatas ou normalizadas foram encontradas
(`exact_duplicate_comments: 0`, `normalized_duplicate_comments: 0`). Não
houve necessidade de decidir entre manter/consolidar/excluir grupos, porque
não há grupos.

### 6. Split reprodutível — feito

`src/rag_hatespeech_ptbr/splits.py` implementa `StratifiedGroupKFold`
(scikit-learn, já uma dependência) agrupado por `link_post` canônico.
Escolha: usar a implementação de referência do scikit-learn em vez de
lógica de split manual, porque já resolve estratificação + agrupamento de
forma testada e citável.

### 7. Baseline determinístico — feito

`src/rag_hatespeech_ptbr/baseline.py`: `TfidfVectorizer` +
`LogisticRegression`, ambos do scikit-learn (já dependência do projeto —
nenhuma dependência nova). Ajustado apenas no treino, avaliado na
validação por padrão. Ver 6_guia_codigo.md.

Justificativa de escolha: um baseline TF-IDF + regressão logística é
determinístico, rápido, não depende de rede/API, e é o tipo de baseline que
a literatura de discurso de ódio usa como piso de comparação (citado no
próprio PCC, seção 1.1, como "abordagens estatísticas anteriores"). Serve
para responder ao objetivo específico "analisar as limitações de
classificadores textuais" com um número real, não apenas uma afirmação
teórica.

### 8. Recuperação (retrieval) — planejado

Ainda não implementado. Provedor de embeddings: Gemini (12_decisoes.md,
"Provedor de LLM: Gemini"). Desenho pretendido:

- Função `embed(texts: list[str]) -> np.ndarray`, isolada atrás de uma
  interface, para poder trocar o provedor de embeddings sem reescrever o
  resto do pipeline.
- Índice restrito a `split == "train"` — nunca `validation` ou `test`.
- Similaridade de cosseno para ranquear os `top_k` exemplos mais próximos,
  com seus `rationales_annotator1`/`rationales_annotator2`.
- Banco vetorial: **tentar Pinecone primeiro** (12_decisoes.md, "Banco
  vetorial: tentar Pinecone primeiro"). O plano gratuito (2 GB, até 300 mil
  vetores, 2M unidades de escrita e 1M de leitura por mês, sem cartão)
  comporta os 5.608 itens de treino sem aperto algum. Se surgir qualquer
  obstáculo real na conta gratuita, cair para uma implementação local
  (`sklearn.metrics.pairwise.cosine_similarity` sobre uma matriz densa em
  memória — também totalmente viável nessa escala) e registrar a
  substituição em 12_decisoes.md, com a justificativa de que os vizinhos
  mais próximos retornados seriam os mesmos nos dois casos.

### 9. Armazenamento/indexação — planejado, mesma decisão do item 8

Se o índice ficar em Pinecone, o armazenamento é gerenciado pelo próprio
serviço (basta guardar o `index name`/configuração usada, para
reprodutibilidade, em `outputs/experiments/`). Se cair para a alternativa
local, "armazenar o índice" significa persistir a matriz de embeddings do
treino (ex.: `.npy` ou `.parquet`) mais os IDs correspondentes em
`data/interim/` ou `outputs/`, com hash da matriz igual ao já feito para o
dataset bruto e o manifesto de split.

### 10. Prompts — não iniciado

Nenhum prompt foi escrito ou testado. Alvo: API do Gemini. Recomendação de
design: escrever os templates de prompt como arquivos de texto versionados
(não *strings* embutidas no código), para que mudanças de prompt fiquem
visíveis no histórico do Git — importante porque o protocolo proíbe
ajustar o prompt com base no resultado do teste final.

### 11. Execução de modelos (LLM) — não iniciado, interface a definir

Recomendação: uma interface mínima
(`class Classifier: def predict(self, comment: str, context: list[Example]) -> Prediction`)
com pelo menos duas implementações: `BaselineClassifier` (já existe,
efetivamente, como `baseline.py`) e uma futura `GeminiClassifier`. Isso
permite reaproveitar o mesmo código de avaliação (`metrics.py`) para todos
os modelos, incluindo baseline, Gemini sem RAG e Gemini com RAG — em vez de
reimplementar cálculo de métricas a cada novo experimento.

Orquestração: **tentar LangChain primeiro** (12_decisoes.md, "Orquestração:
tentar LangChain primeiro"); cair para chamadas diretas à API do Gemini com
código próprio se a abstração atrapalhar mais do que ajudar.

Nenhuma chamada de API foi feita ainda. Antes do cliente real, implementar
um modo `dry-run`/mock (resposta determinística fixa, sem rede), para
permitir testar o pipeline de ponta a ponta sem custo. Rodar os
experimentos reais primeiro na camada gratuita da API do Gemini; só
habilitar faturamento se os limites de taxa da camada gratuita não
comportarem a escala do projeto (ver 12_decisoes.md).

### 12. Avaliação — feito, como módulo reutilizável

`src/rag_hatespeech_ptbr/metrics.py`: calcula F1/precisão/revocação da
classe ofensiva, macro-F1 e matriz de confusão a partir de rótulos
verdadeiros e previstos. Usado pelo baseline agora; deve ser reaproveitado
sem alteração pelos experimentos de Gemini e RAG futuros.

### 13. Análise por sarcasmo — não iniciado, depende da anotação

Depende inteiramente da anotação humana de sarcasmo (piloto + teste), que
ainda não começou. Nenhum código de análise estratificada existe ainda
porque não há rótulos de sarcasmo para analisar. O módulo de métricas
(item 12) foi desenhado para aceitar um filtro por subconjunto (ex.:
`evaluate(y_true, y_pred, mask=...)`), então a análise por estrato deve
ser um reuso direto, não uma reimplementação.

### 14. Geração de tabelas — feito, padrão estabelecido

Cada script de auditoria/experimento grava um JSON agregado em
`outputs/tables/` ou `outputs/experiments/`, seguindo o mesmo formato dos
relatórios já existentes (`dataset_audit.json`, `split_audit.json`). O
baseline segue o mesmo padrão. Tabelas em formato pronto para
LaTeX/Markdown (para colar direto no TCC) ainda não existem — ficam como
item de trabalho futuro, de baixo risco técnico, a ser feito quando houver
mais de um modelo para comparar (uma tabela com uma linha só não vale o
esforço de gerar automaticamente ainda).

### 15. Geração de figuras — não iniciado

Nenhuma figura de resultado existe (`outputs/figures/` só tem `.gitkeep`).
Não há necessidade ainda: com um único modelo avaliado, não há comparação
visual significativa para plotar. `matplotlib` já é dependência de
desenvolvimento (`pyproject.toml`), então nenhuma dependência nova será
necessária quando chegar a hora.

### 16. Registro de experimentos — feito

`scripts/run_baseline.py` grava um registro JSON por execução em
`outputs/experiments/`, com: identificador do experimento, timestamp,
*hash* do commit Git (se disponível), hashes do dataset e do manifesto de
split usados, hiperparâmetros, *split* avaliado e métricas obtidas. Este
formato deve ser reaproveitado por todos os experimentos futuros (Gemini,
RAG) para manter um histórico comparável e auditável de todas as
execuções, incluindo quantas vezes o conjunto de teste foi avaliado (ver
8_auditoria_vazamento.md). Para chamadas de API, adicionar também campos
de custo e latência ao registro.

### 17. Interface de avaliação de explicabilidade — planejado, adiado

Formulário/interface simples + banco Postgres + hospedagem gratuita (ex.:
Vercel), para coletar opiniões de um grupo de conhecidos sobre a qualidade
das justificativas geradas pelo RAG (12_decisoes.md, "Direção aceita para a
análise de explicabilidade"). Implementação adiada até existirem
justificativas reais para avaliar — não é um componente do pipeline de
classificação em si, é uma ferramenta de coleta de dados para a análise
qualitativa do objetivo específico 5 do PCC.

### 18. Serviço FastAPI — planejado, incluído no escopo

Serviço HTTP de inferência ao vivo, como descrito no PCC. Decisão: incluir
no escopo do TCC (12_decisoes.md, "Incluir FastAPI no escopo"),
implementado depois que o pipeline de experimentos estiver funcionando —
não é necessário para validar a hipótese de pesquisa, mas há tempo
disponível e valor como demonstração na defesa.

## Dependências novas: plano de adoção

Diferente da avaliação anterior (que classificava Pinecone/LangChain como
"rejeitados por enquanto"), a decisão atual é **tentar as duas primeiro**,
já que nenhuma delas impõe custo obrigatório na escala deste projeto —
ver 12_decisoes.md para o raciocínio completo de cada uma.

| Dependência | Para quê será usada | Plano |
|---|---|---|
| Gemini API | Geração (classificação com/sem RAG) e possivelmente *embeddings* | Provedor escolhido. Começar na camada gratuita da API; modo *dry-run* antes de qualquer chamada real. |
| Pinecone | Banco vetorial gerenciado para o índice de recuperação | Tentar primeiro (plano gratuito comporta os 5.608 itens de treino com folga). Se travar, substituir por implementação local, com justificativa de equivalência de resultado. |
| LangChain | Orquestração de *prompt* + *retrieval* | Tentar primeiro (gratuito; a única barreira possível é complexidade, não custo). Se atrapalhar mais do que ajudar, substituir por chamadas diretas à API. |
| FastAPI | Serviço HTTP de inferência | Incluído no escopo, implementado por último, depois que o pipeline de experimentos estiver pronto. |
| sentence-transformers / modelos locais de *embedding* | Alternativa gratuita a APIs pagas de *embedding* | Não é mais o plano principal (Gemini foi escolhido), mas permanece como alternativa se a cota gratuita do Gemini for insuficiente. |

## Próximo incremento de implementação recomendado

1. Interface `Classifier` + modo *dry-run* do cliente Gemini.
2. Baseline Gemini sem recuperação, rodado em pequena escala na validação
   (camada gratuita da API).
3. Função de *embedding* + tentativa de índice no Pinecone, restrito ao
   treino.
4. RAG completo primeiro na validação (para calibrar `top_k` e prompt),
   via LangChain como primeira tentativa.
5. Anotação de sarcasmo (pode ocorrer em paralelo às etapas 1–4, pois é
   inteiramente humana e não depende delas).
6. Avaliação final única no teste.
7. Serviço FastAPI e interface de avaliação de explicabilidade, depois
   que 1–6 estiverem prontos.
