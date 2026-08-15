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

### Próximos passos (ordem recomendada)

1. Revisar este relatório e 3_checklist_revisao.md (ver
   2_ordem_revisao.md para a ordem exata).
2. Agendar e executar o piloto de anotação de sarcasmo (100 itens da
   validação) — pode acontecer em paralelo aos itens abaixo.
3. Implementar a interface de embeddings + tentativa de índice no
   Pinecone, restrito ao treino.
4. Implementar o baseline do Gemini sem recuperação (com modo *dry-run*
   primeiro).
5. Implementar RAG completo, calibrado na validação, via LangChain como
   primeira tentativa.
6. Só então: avaliação final única no teste, com `--confirm-final-test-run`.
7. Serviço FastAPI e interface de avaliação de explicabilidade.
