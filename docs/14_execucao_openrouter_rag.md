# Execução de LLM e RAG — 2 de outubro de 2026

Este é o ponto de retomada atual. OpenRouter substitui a integração direta com
Gemini; o modelo de geração ainda será escolhido antes da primeira inferência.

## Fase atual

Dados auditados, splits congelados (5.608 treino / 706 validação / 686 teste) e
baseline TF-IDF + regressão logística concluídos. O F1 ofensivo de 0,7646 é da
validação do baseline estatístico, não do LLM ou do RAG.

O autor anotou toda a validação: 627 não sarcásticos, 18 sarcásticos e 61 incertos.
As planilhas serão preservadas como entregues, por decisão do autor. Evidências
e notas incompletas e concordância limitada do piloto são limitações a relatar,
não impedimentos para implementar o sistema. A anotação de sarcasmo do teste
permanece pendente. Análises por sarcasmo na validação são exploratórias e não
equivalem à análise final do teste prevista no PCC.

Pipeline implementado e índice semântico real de treino gerado. A chave OpenRouter
será usada depois; nenhuma classificação real por API foi feita nesta etapa.

## Componentes

- Cliente OpenRouter para geração estruturada e embeddings, com limite de
  requisições, uso, latência e registro dos modelos/provedores retornados.
- Embeddings semânticos multilíngues locais ou via OpenRouter; estes últimos
  têm cache SQLite por modelo, configuração do provedor e hash do texto.
- Índice local exato por cosseno, somente com os IDs de treino. Um comentário
  inteiro por exemplo; sem chunking. Rationales ausentes não são inventados.
- Mesmo LLM e prompt para `baseline`, `rag` e `rag_no_rationales`; a ablação
  retira somente os rationales dos mesmos exemplos recuperados.
- Execuções retomáveis, com configuração/hashes, IDs, predições, vizinhos e métricas.
- FastAPI local: `/health`, `/classify` e documentação interativa `/docs`.

Pinecone e LangChain ainda não foram integrados. O índice local é a referência
funcional desta etapa; migrar/adicionar serviços exige preservar os controles
experimentais. Não houve tentativa de conta Pinecone nem comparação empírica
entre os índices. As decisões históricas não devem ser lidas como implementação.

## Verificar sem chave

```bash
source .venv/bin/activate
python -m pip install -e '.[dev,api]'
python -m scripts.build_retrieval_index data/raw/HateBRXplain.csv
python -m scripts.run_llm_experiment data/raw/HateBRXplain.csv \
  --modes baseline rag rag_no_rationales --limit 20 \
  --run-dir outputs/experiments/smoke_dry_run_v1
```

Sem `--live`, os vetores são hashing e a classificação é uma fixture técnica:
`scientific_result=false`, `metrics=null`. Não usar como desempenho do RAG.
O índice de simulação tem nome diferente do índice real. Criar índice recusa
sobrescrever um arquivo existente: reutilize-o ou escolha outro `--index`.

## Índice semântico real, sem OpenRouter

Instalar Torch CPU primeiro evita baixar bibliotecas CUDA sem necessidade:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[local-embeddings]'
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m scripts.build_retrieval_index \
  data/raw/HateBRXplain.csv --live --embedding-backend local
```

Aqui, `--live` significa embeddings reais; `--embedding-backend local` dispensa
chave/API. Os pesos públicos são baixados na primeira execução para
`data/interim/model_cache/`. O índice fica em `data/interim/train_index.npz`, com
auditoria ao lado; ambos fora do Git. Este índice já foi gerado nesta sessão.

Modelo inicial: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`,
revisão `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, 384 dimensões, CPU,
normalização L2 e recuperação por cosseno. Limite original: 128 tokens.
34 dos 5.608 textos de treino ultrapassam esse limite na tokenização; a auditoria
registra a truncagem dos embeddings, mas os comentários originais permanecem
inteiros no prompt. Avaliar esse efeito na validação antes da configuração final.
Fonte: [modelo oficial](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).

Consulta local com texto inventado, sem LLM:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m scripts.query_retrieval \
  --comment 'Discordo da proposta apresentada.'
```

Mostra IDs e similaridades. Pode consultar também o índice dry-run por `--index`.

Comparação completa do fluxo com recuperação semântica real e LLM simulado:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m scripts.run_llm_experiment \
  data/raw/HateBRXplain.csv --semantic-retrieval \
  --modes baseline rag rag_no_rationales --limit 20 \
  --run-dir outputs/experiments/smoke_semantic_v1
```

Continua sem chave e sem métricas científicas; testa os vizinhos reais e as
três condições antes de chamar o modelo pela API.

## Preparar OpenRouter para depois

Copie `.env.example` para `.env` e preencha a chave somente localmente. O `.env`
é ignorado pelo Git. `OPENROUTER_CHAT_MODEL` foi deixado vazio até escolher o modelo.

```bash
python -m scripts.list_openrouter_models --free-only --structured-only
python -m scripts.list_openrouter_models --embeddings
```

O catálogo público não usa chave nem faz inferência. Uma chave gratuita não torna
todos os modelos gratuitos. Para inferência gratuita, escolha um ID fixo `:free`
compatível com os parâmetros/JSON Schema. Não usar `openrouter/free` ou
`openrouter/auto`: esses roteadores podem escolher LLMs diferentes entre condições.

Depois de escolher o modelo e configurar a chave, a primeira execução será pequena:

```bash
python -m scripts.run_llm_experiment data/raw/HateBRXplain.csv \
  --live --embedding-backend local --modes baseline rag --limit 5 \
  --run-dir outputs/experiments/openrouter_validation_5_v1 \
  --annotations data/annotations/sarcasm_pilot.xlsx \
                data/annotations/sarcasm_pilot_batch2.xlsx
```

São até dez chamadas de geração para cinco comentários em dois modos. Conferir
suporte do endpoint, recusas e cota real. O cliente solicita
`provider.data_collection=deny` e `require_parameters=true`. Sem rota compatível,
falha explicitamente; não troca automaticamente por um modelo pago nem relaxa
silenciosamente essa configuração.
Fontes: [roteamento](https://openrouter.ai/docs/guides/routing/provider-selection),
[saída estruturada](https://openrouter.ai/docs/guides/features/structured-outputs),
[limites](https://openrouter.ai/docs/api_reference/limits).

Embeddings via OpenRouter são outra opção: configurar `EMBEDDING_BACKEND=openrouter`
e `OPENROUTER_EMBEDDING_MODEL`, construir outro `--index` e consultar sempre com a
mesma configuração. Essa alternativa pode consumir créditos.
Fonte: [embeddings](https://openrouter.ai/docs/api/api-reference/embeddings/create-embeddings).

## Saídas e retomada

| Arquivo em `--run-dir` | Conteúdo |
|---|---|
| `config.json` | Modelos, prompt, hashes, versões, modos e IDs avaliados |
| `environment.json` | Início da execução e estado do Git |
| `predictions.jsonl` | Classificação, justificativa e vizinhos por ID/modo |
| `api_calls.json` | Uso, latência e modelo/provedor retornados, sem prompts/chave |
| `summary.json` | Estado, métricas e custos informados |

Repetir o comando com o mesmo `--run-dir` retoma as predições faltantes. Alterou
prompt/modelo/índice/anotações/código? Use outro diretório. Não há repetição
automática: um timeout pode ocorrer depois que o servidor processou/cobrou o pedido.
Limite de chamadas não é teto monetário. Custo ausente permanece desconhecido,
nunca vira zero; use também os controles de crédito da conta.

Métricas reais são calculadas após todos os pares previstos terminarem. O F1
principal inclui todos os itens, independentemente do sarcasmo. Com anotações,
os estratos `0`, `1` e `2` são reportados separadamente, sem converter incertos.
Rótulos de sarcasmo nunca entram na seleção, no índice ou no prompt.

## Demonstração

```bash
python -m uvicorn rag_hatespeech_ptbr.api:app --host 127.0.0.1 --port 8000
```

Abrir `http://127.0.0.1:8000/docs`. Por padrão usa índice dry-run/fixture.
Para inferência real, iniciar depois com `RAG_LIVE=1`, chave e modelo configurados.
A demonstração ainda não tem autenticação/limite por usuário: mantenha-a local.
Os pedidos dessa interface não substituem experimentos registrados pela CLI.

## Próximas entregas

1. Escolher um LLM fixo e testar cinco itens reais com e sem RAG.
2. Ampliar na validação, comparar `top_k`/ablação e analisar erros/custos.
3. Congelar a configuração e combinar com o orientador o alcance da análise
   por sarcasmo, considerando poucos casos e a anotação do teste pendente.
4. Avaliar uma vez no teste completo: `--split test --limit 0
   --confirm-final-test-run`. Acesso real ao teste fica registrado no ledger.
5. Produzir comparações pareadas com reamostragem por publicação, tabelas,
   figuras, avaliação de justificativas e redação final do TCC.

Temperatura zero e semente solicitada não garantem determinismo da API;
modelos/provedores retornados são registrados para documentar essa limitação.
