# Relatório de sessões

Log cronológico do que foi feito em cada sessão de trabalho no projeto.
Cada sessão nova é acrescentada como uma seção `## Sessão de AAAA-MM-DD`
abaixo, sem apagar as anteriores.

## Sessão de 2026-08-15

### Parte 1 — baseline determinístico e documentação inicial

**Arquivos alterados:**
- `README.md` — nova seção de estado, instruções de uso do baseline.
- 10_plano_dados.md — tabela de progresso atualizada, item 1 de "O que vem
  depois" marcado concluído.
- 12_decisoes.md — duas novas decisões (baseline determinístico, política
  de execução única no teste).
- 9_protocolo_pesquisa.md — "pontos abertos" corrigidos (a pergunta de
  pesquisa já estava congelada pelo PCC; isso não estava refletido antes).

**Arquivos criados — documentação:**
- Os arquivos numerados que compõem `docs/` hoje (ver README para o
  índice).

**Arquivos criados — código:**
- `src/rag_hatespeech_ptbr/data.py` — carregamento unificado do dataset +
  manifesto de split, com verificação de hash.
- `src/rag_hatespeech_ptbr/metrics.py` — métricas de classificação
  compartilhadas (F1, precisão, revocação, macro-F1, matriz de confusão),
  com suporte a `mask` para a futura análise por estrato de sarcasmo.
- `src/rag_hatespeech_ptbr/baseline.py` — baseline TF-IDF + regressão
  logística.
- `scripts/run_baseline.py` — CLI que treina, avalia e registra o
  experimento, com bloqueio técnico contra avaliação silenciosa no teste.

**Arquivos criados — testes:**
- `tests/test_data.py`, `tests/test_metrics.py`, `tests/test_baseline.py`
  (15 testes novos).

**Arquivos criados — outputs:**
- `outputs/tables/baseline_validation.json` (versionado, canônico).
- `outputs/experiments/tfidf_logreg_baseline-e7790f76.json` (não
  versionado, `.gitignore` já cobria o padrão).

**Nenhuma dependência nova** foi adicionada a `pyproject.toml` — o baseline
usa apenas `scikit-learn`, já presente.

### Resultados obtidos

Baseline determinístico (TF-IDF + regressão logística), treinado em 5.608
itens de treino, avaliado nos 706 itens de validação:

| Métrica | Valor |
|---|---|
| F1 (classe ofensiva) | 0,7646 |
| Precisão (classe ofensiva) | 0,7701 |
| Revocação (classe ofensiva) | 0,7592 |
| Macro-F1 | 0,7663 |
| Matriz de confusão | `[[273, 80], [85, 268]]` |

Produzidos executando `python -m scripts.run_baseline
data/raw/HateBRXplain.csv` contra o dataset local. Reproduzível com o mesmo
comando; ver `outputs/tables/baseline_validation.json` para o registro
completo (hashes, hiperparâmetros, timestamp, commit).

Não avaliado no teste — de propósito, ver decisão em 12_decisoes.md
(2026-08-15).

### Parte 2 — reorganização da documentação e decisões de arquitetura

Depois da primeira leva de documentos, o autor pediu uma reorganização:
menos arquivos, nomes numerados (`N_nome_do_arquivo.md`), sem linguagem que
soe como "a IA não pode fazer isso" (o documento final é sempre revisado
pelo autor antes de ir para o repositório remoto), e resolveu, na mesma
conversa, a maior parte das decisões que estavam em aberto. Mudanças desta
parte:

- Documentação reorganizada de 15 arquivos em `docs/` para 13, numerados e
  com nomes em `snake_case`. Arquivos que descreviam a mesma coisa de
  ângulos diferentes foram unidos: as notas históricas de revisão do PCC
  entraram em 12_decisoes.md; a matriz de rastreabilidade PCC/TCC e os
  critérios de "TCC ideal" entraram em 13_guia_escrita_tcc.md.
- Linguagem de auto-referência removida de todos os arquivos (ex.: "eu como
  IA não posso fazer X" virou "X ainda não foi feito" / "X é uma tarefa
  humana, ainda pendente").
- Decisões resolvidas nesta conversa, todas registradas em 12_decisoes.md
  com data 2026-08-15: provedor de LLM (Gemini, com ressalva sobre
  faturamento da API vs. assinatura de consumo), banco vetorial (tentar
  Pinecone primeiro — plano gratuito confirmado como suficiente por
  pesquisa), orquestração (tentar LangChain primeiro), escopo do FastAPI
  (incluído), esclarecimento do rótulo `incerto` de sarcasmo, segundo
  anotador (confirmado — um amigo do autor), proposta de magnitude para a
  hipótese quantitativa (cálculo de poder estatístico, pendente de
  confirmação do orientador), direção da análise de explicabilidade
  (formulário + Postgres + hospedagem gratuita, implementação adiada),
  fixação de versão do pandas (adiada para o fim), versionamento do
  manifesto de split (decidido versionar), e confirmação sobre citar
  exemplos do dataset no texto do TCC (permitido pela licença CC BY-NC
  4.0, verificado por pesquisa).
- `data/processed/split_manifest.csv` passou a ser versionado
  (`.gitignore` ajustado com uma exceção específica).

### Problemas restantes

- **Objetivo específico 5 do PCC** (explicabilidade das justificativas)
  ainda não tem desenho experimental totalmente fechado — a direção foi
  aceita (formulário de avaliação), mas os detalhes (perguntas, tamanho de
  amostra) dependem de o RAG já estar produzindo saídas.
- **Versão do pandas** não fixada exatamente (`>=2.2,<3`), ao contrário do
  scikit-learn (`==1.9.0`) — decisão consciente de adiar para o fim.
- **Cota gratuita do Gemini** ainda não verificada para o modelo específico
  que será usado — depende de decisão tomada no momento da implementação.

### Parte 3 — ferramenta de preparação do piloto de sarcasmo

O autor pediu instruções concretas de como fazer a anotação manual
(qual planilha, onde olhar, onde anotar, como enviar). Em vez de só
explicar, foi implementada uma ferramenta:

- `src/rag_hatespeech_ptbr/sarcasm_pilot.py` — sorteio determinístico de
  100 itens da validação (50/50 por `offensive_label`, nunca do teste,
  semente 42) e de um subconjunto de ~15 itens para o segundo anotador
  (mesmos `id`s de parte do piloto, semente 43). `offensive_label` nunca
  entra na planilha de anotação.
- `scripts/prepare_sarcasm_pilot.py` — CLI que gera as duas planilhas
  `.xlsx` (com menu suspenso para os rótulos `0`/`1`/`2` e para
  `review_status`, cabeçalho congelado, texto do comentário com quebra de
  linha) e um relatório agregado sem texto.
- `tests/test_sarcasm_pilot.py` — 6 testes novos, incluindo a checagem de
  vazamento mais importante aqui: nenhum `id` sorteado pode vir de
  `train`/`test`.

Executado nesta sessão contra o dataset real:
`data/annotations/sarcasm_pilot.xlsx` (100 itens),
`data/annotations/sarcasm_pilot_second_annotator.xlsx` (15 itens) — os
dois ficam locais, fora do Git, porque contêm texto real de comentários.
`outputs/tables/sarcasm_pilot_audit.json` (versionado, só ids e
contagens) confirma a amostra e é reproduzível — reexecutado uma segunda
vez, saiu byte-idêntico.

Passo a passo completo de como preencher e o que fazer com a planilha do
segundo anotador está em 11_diretrizes_anotacao.md, seção "Passo a passo
do piloto".

## Sessão de 2026-08-15 (continuação) — resultado real do piloto

O autor preencheu as duas planilhas do piloto (100 itens + subconjunto de
15 do segundo anotador) e trouxe o resultado de volta. Números reais,
lidos das planilhas preenchidas:

- Anotador principal (100 itens): 84× `0`, 15× `2` (incerto), 1× `1`
  (sarcástico).
- Segundo anotador (15 itens, subconjunto): 13× `0`, 2× `1`, 0× `2`.
- Concordância nos 15 itens em comum: 80% bruta, Cohen's kappa 0,196
  (calculado por `scripts/check_annotation_agreement.py`, novo nesta
  sessão, registrado em `outputs/tables/sarcasm_annotation_agreement.json`).

Interpretação e decisão registradas em 12_decisoes.md ("Resultado real do
piloto de sarcasmo e risco de poder estatístico"): taxa de sarcasmo
claramente identificado baixa (~1%), coerente com o corpus ser majoritariamente
ofensa direta, não irônica. Isso não afeta o experimento principal (RAG
vs. Gemini sem RAG sobre `offensive_label`), só a análise secundária por
estrato sarcástico, que fica sob risco de baixo poder estatístico —
registrado como decisão pendente (D12 em 5_decisoes_pendentes.md, precisa
de confirmação do orientador se a taxa continuar baixa).

Ação tomada: implementado `scripts/expand_sarcasm_sample.py` (amostragem
aleatória adicional dentro da validação, balanceada por ofensividade,
excluindo ids já em circulação — nunca curadoria manual, nunca usando
*rationales* de ofensividade como pista de sarcasmo). Rodado contra o
dataset real: gerou `data/annotations/sarcasm_pilot_batch2.xlsx` com os
606 itens restantes da validação (303 por classe). Confirmado que nenhum
id vem de treino/teste, e que a execução é reprodutível.

### Próximos passos (ordem recomendada)

1. Revisar este relatório e 3_checklist_revisao.md (ver
   2_ordem_revisao.md para a ordem exata).
2. Preencher `data/annotations/sarcasm_pilot_batch2.xlsx` (606 itens, no
   ritmo que for possível) para reduzir a incerteza da taxa de sarcasmo
   antes de decidir D12.
3. Rodar `scripts/check_annotation_agreement.py` de novo sempre que
   houver anotação nova do segundo anotador para acompanhar a
   concordância.
4. Levar o achado (taxa baixa de sarcasmo) e a decisão D12 para o
   orientador.
5. Implementar a interface de embeddings + tentativa de índice no
   Pinecone, restrito ao treino.
6. Implementar o baseline do Gemini sem recuperação (com modo *dry-run*
   primeiro).
7. Implementar RAG completo, calibrado na validação, via LangChain como
   primeira tentativa.
8. Só então: avaliação final única no teste, com `--confirm-final-test-run`.
9. Serviço FastAPI e interface de avaliação de explicabilidade.

## Sessão de 2026-10-02 — Implementação OpenRouter/RAG

- Validação anotada integralmente (706 comentários). O autor decidiu preservar
  o trabalho entregue e seguir para a implementação, sem nova edição de planilha.
- Cliente OpenRouter de geração/embeddings, prompt versionado e parsing estrito;
  embeddings locais e cache SQLite para a alternativa remota; índice de treino
  por cosseno; baseline LLM, RAG e ablação comparáveis e retomáveis.
- Índice semântico real gerado localmente: 5.608 × 384, modelo/revisão fixados,
  34 textos acima do limite de tokens, dataset/manifesto verificados por hash.
  Consulta de recuperação executada sem usar chave ou inferência OpenRouter.
- Serviço FastAPI local implementado e testes do pipeline verificados.
- Chave guardada para depois. O próximo passo é escolher um modelo fixo e rodar
  cinco comentários da validação com e sem RAG; teste final ainda reservado.
- Comandos e estado atual em `14_execucao_openrouter_rag.md`. Pinecone,
  LangChain, análise estatística final e redação permanecem etapas posteriores.
