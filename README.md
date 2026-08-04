# rag-hatespeech-ptbr

Projeto de TCC sobre classificação de linguagem ofensiva em português brasileiro,
com foco na avaliação de comentários sarcásticos e no uso de exemplos recuperados
do HateBRXplain.

> Status: início do TCC. O escopo foi consolidado na versão final do PCC de
> 22 de julho de 2026; a fase atual é a preparação reprodutível dos dados.

## Objetivo

Comparar um LLM sem recuperação com uma abordagem RAG que recupera exemplos e
*rationales* de ofensividade, medindo o desempenho no teste completo e,
separadamente, nos comentários anotados como sarcásticos.

Os *rationales* do HateBRXplain justificam ofensividade; eles não são rótulos de
sarcasmo. O projeto mantém `offensive_label` como alvo e prevê uma anotação
adicional de sarcasmo para análise estratificada.

## Estrutura

```text
data/         dados locais (arquivos brutos e derivados não são versionados)
docs/         protocolo, decisões e guia de anotação
notebooks/    análises exploratórias
outputs/      tabelas, figuras e execuções experimentais
scripts/      comandos reprodutíveis do pipeline
src/          código-fonte reutilizável
tests/        testes automatizados
```

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

O manifesto local fica em `data/processed/split_manifest.csv` e contém somente
`id`, `fold` e `split`. Seu hash e as estatísticas agregadas ficam registrados em
`outputs/tables/split_audit.json`.

## Fase atual: dados

Antes de implementar embeddings, banco vetorial ou integração com LLM, o projeto
deve concluir quatro entregas:

1. obter a versão oficial do HateBRXplain e registrar origem, versão, licença e
   integridade do arquivo, sem versionar os comentários;
2. documentar o esquema real, classes, valores ausentes, *rationales* e duplicatas;
3. gerar identificadores estáveis e partições reprodutíveis sem vazamento;
4. produzir um relatório agregado que permita ao orientador aprovar a base antes
   da indexação e da anotação de sarcasmo.

O plano, os artefatos esperados e os critérios de conclusão estão em
[`docs/data-preparation-plan.md`](docs/data-preparation-plan.md).

### Estado em 4 de agosto de 2026

- Aquisição e proveniência: concluídas.
- Contrato e auditoria: concluídos, com todos os controles aprovados.
- Partições agrupadas por publicação: congeladas e aprovadas.
- Próxima etapa: calibrar o guia de sarcasmo com um piloto da validação.

## Cuidados

- Não versionar o dataset bruto, comentários individuais ou credenciais.
- Não publicar amostras do corpus antes de confirmar sua licença e seus termos.
- Preservar emojis, caixa, pontuação e repetições na inspeção inicial.
- Definir a divisão antes da anotação de sarcasmo.
- Construir o índice de recuperação somente com o treino.
- Tratar a pré-anotação por IA como apoio; o rótulo final é humano.

O corpus contém linguagem potencialmente ofensiva. Consulte a licença e os termos
do dataset oficial antes de redistribuir conteúdo.
