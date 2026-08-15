# rag-hatespeech-ptbr

Projeto de TCC sobre classificação de linguagem ofensiva em português brasileiro,
com foco na avaliação de comentários sarcásticos e no uso de exemplos recuperados
do HateBRXplain.

> Status: início do TCC. O escopo foi consolidado na versão final do PCC de
> 22 de julho de 2026; a fase atual é a preparação reprodutível dos dados e
> a implementação do primeiro baseline.

## Objetivo

Comparar um LLM sem recuperação com uma abordagem RAG que recupera exemplos e
*rationales* de ofensividade, medindo o desempenho no teste completo e,
separadamente, nos comentários anotados como sarcásticos.

Os *rationales* do HateBRXplain justificam ofensividade; eles não são rótulos de
sarcasmo. O projeto mantém `offensive_label` como alvo e prevê uma anotação
adicional de sarcasmo para análise estratificada.

## Estrutura

```text
data/         dados locais (arquivos brutos e derivados não são versionados,
              exceto o manifesto de split)
docs/         protocolo, decisões, guia de anotação e documentação técnica
notebooks/    análises exploratórias
outputs/      tabelas, figuras e execuções experimentais
scripts/      comandos reprodutíveis do pipeline
src/          código-fonte reutilizável
tests/        testes automatizados
```

### Índice de `docs/`

Os arquivos são numerados na ordem sugerida de leitura para quem está
retomando o projeto. `1_relatorio_sessao.md` é o ponto de partida.

| # | Arquivo | Conteúdo |
|---|---|---|
| 1 | [`1_relatorio_sessao.md`](docs/1_relatorio_sessao.md) | Log cronológico do que foi feito em cada sessão de trabalho |
| 2 | [`2_ordem_revisao.md`](docs/2_ordem_revisao.md) | Ordem recomendada para revisar o estado atual do projeto |
| 3 | [`3_checklist_revisao.md`](docs/3_checklist_revisao.md) | Checklist de revisão (crítico/importante/mecânico) |
| 4 | [`4_estado_projeto.md`](docs/4_estado_projeto.md) | Estado atual do projeto: o que está feito, pendente e bloqueado |
| 5 | [`5_decisoes_pendentes.md`](docs/5_decisoes_pendentes.md) | O que ainda depende de decisão do autor/orientador |
| 6 | [`6_guia_codigo.md`](docs/6_guia_codigo.md) | Passo a passo do código, arquivo por arquivo |
| 7 | [`7_arquitetura_pipeline.md`](docs/7_arquitetura_pipeline.md) | Desenho dos componentes do pipeline, feitos e planejados |
| 8 | [`8_auditoria_vazamento.md`](docs/8_auditoria_vazamento.md) | Auditoria de mecanismos de vazamento (leakage) |
| 9 | [`9_protocolo_pesquisa.md`](docs/9_protocolo_pesquisa.md) | Protocolo experimental: pergunta, métricas, controles |
| 10 | [`10_plano_dados.md`](docs/10_plano_dados.md) | Plano de preparação dos dados, etapa a etapa |
| 11 | [`11_diretrizes_anotacao.md`](docs/11_diretrizes_anotacao.md) | Guia de anotação de sarcasmo |
| 12 | [`12_decisoes.md`](docs/12_decisoes.md) | Registro histórico de todas as decisões do projeto |
| 13 | [`13_guia_escrita_tcc.md`](docs/13_guia_escrita_tcc.md) | Guia para escrever o TCC: rastreabilidade PCC→TCC e critérios de qualidade |

## Primeiros passos

Requer Python 3.11 ou superior.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Baixe a revisão fixada do HateBRXplain em `data/raw/`, sem versioná-la:

```bash
curl -L --fail \
  "https://raw.githubusercontent.com/franciellevargas/HateBR/0ac05461f17c4d7b7655fbee0391ec325e61a83b/dataset/HateBRXplain.csv" \
  -o data/raw/HateBRXplain.csv
```

Então execute a auditoria:

```bash
python scripts/inspect_dataset.py data/raw/HateBRXplain.csv
```

O script verifica origem, hash, esquema, classes, IDs, duplicatas e presença de
*rationales*. Ele não inclui comentários ou links individuais na saída e grava o
relatório agregado em `outputs/tables/dataset_audit.json`.

Gere as partições congeladas:

```bash
python -m scripts.create_splits data/raw/HateBRXplain.csv
```

O manifesto fica em `data/processed/split_manifest.csv` (versionado — contém
somente `id`, `fold` e `split`, sem texto). Seu hash e as estatísticas
agregadas ficam registrados em `outputs/tables/split_audit.json`.

Execute o baseline determinístico (treina no treino, avalia por padrão na
validação):

```bash
python -m scripts.run_baseline data/raw/HateBRXplain.csv
```

O relatório fica em `outputs/tables/baseline_validation.json`. Avaliar no
conjunto de teste exige `--split test --confirm-final-test-run` e fica
registrado em `outputs/experiments/test_evaluations_ledger.jsonl` — ver
[`docs/12_decisoes.md`](docs/12_decisoes.md) (2026-08-15) para a justificativa.

## Fase atual: dados e primeiro baseline

Antes de implementar embeddings, banco vetorial ou integração com LLM, o projeto
concluiu quatro entregas:

1. obter a versão oficial do HateBRXplain e registrar origem, versão, licença e
   integridade do arquivo, sem versionar os comentários;
2. documentar o esquema real, classes, valores ausentes, *rationales* e duplicatas;
3. gerar identificadores estáveis e partições reprodutíveis sem vazamento;
4. produzir um relatório agregado que permita ao orientador aprovar a base antes
   da indexação e da anotação de sarcasmo.

O plano, os artefatos esperados e os critérios de conclusão estão em
[`docs/10_plano_dados.md`](docs/10_plano_dados.md).

### Estado em 15 de agosto de 2026

- Aquisição e proveniência: concluídas.
- Contrato e auditoria: concluídos, com todos os controles aprovados.
- Partições agrupadas por publicação: congeladas e aprovadas.
- Baseline determinístico (TF-IDF + regressão logística): implementado e
  avaliado na validação — F1 da classe ofensiva de 0,7646
  (`outputs/tables/baseline_validation.json`). Ainda não avaliado no
  teste, de propósito (ver [`docs/12_decisoes.md`](docs/12_decisoes.md), 2026-08-15).
- Decisões de arquitetura tomadas: Gemini como provedor de LLM/embeddings;
  Pinecone e LangChain serão tentados primeiro, com alternativa local se
  necessário; FastAPI incluído no escopo (ver
  [`docs/12_decisoes.md`](docs/12_decisoes.md)).
- Próxima etapa: executar o piloto de anotação de sarcasmo (tarefa humana,
  não delegável) e implementar o baseline de LLM sem recuperação.
- Para o estado completo e detalhado do projeto, ver
  [`docs/4_estado_projeto.md`](docs/4_estado_projeto.md).

## Cuidados

- Não versionar o dataset bruto, comentários individuais ou credenciais.
- Não publicar amostras do corpus além do que a licença permite (ver
  [`docs/12_decisoes.md`](docs/12_decisoes.md) para a análise da licença
  CC BY-NC 4.0).
- Preservar emojis, caixa, pontuação e repetições na inspeção inicial.
- Definir a divisão antes da anotação de sarcasmo.
- Construir o índice de recuperação somente com o treino.
- Tratar a pré-anotação por IA como apoio; o rótulo final é humano.

O corpus contém linguagem potencialmente ofensiva. Consulte a licença e os termos
do dataset oficial antes de redistribuir conteúdo.
