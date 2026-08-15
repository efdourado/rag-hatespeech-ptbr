# Passo a passo do código

Objetivo deste documento: permitir reconstruir mentalmente o projeto sem
precisar abrir todos os arquivos primeiro. Leia na ordem — cada seção
assume que você leu as anteriores. Fluxo completo:

```
data/raw/HateBRXplain.csv
  → outputs/tables/dataset_audit.json         (inspect_dataset.py)
  → data/processed/split_manifest.csv          (create_splits.py)
  → outputs/tables/split_audit.json
  → outputs/tables/baseline_validation.json     (run_baseline.py)
  → [ainda não existe: recuperação → RAG → avaliação final]
```

## 0. Como instalar e preparar o ambiente

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Isso instala o pacote `rag_hatespeech_ptbr` (de `src/`) em modo editável,
mais `pandas`, `scikit-learn`, `pytest`, `ruff` e as demais dependências de
`pyproject.toml`. Nenhuma dependência nova foi adicionada nesta sessão.

Baixe o dataset (ver README para o comando `curl` com o hash oficial) em
`data/raw/HateBRXplain.csv`. Sem esse arquivo, nenhum script de dados
funciona — mas todos os testes automatizados funcionam, porque usam dados
sintéticos gerados em memória.

## 1. `scripts/inspect_dataset.py` — auditoria do dataset bruto

**Objetivo:** confirmar que o arquivo baixado é exatamente a revisão
esperada do HateBRXplain, documentar seu esquema real e detectar problemas
(duplicatas, valores ausentes, conflitos de rótulo) sem nunca expor texto
de comentários no relatório.

**Função principal:** `build_report(frame, source, source_sha256=None) -> dict`.

**Entrada:** um `pandas.DataFrame` já carregado (por `load_dataset()`, que
aceita CSV/JSON/JSONL) e o `Path` do arquivo original.

**Transformação:**
1. Calcula hash SHA-256 do arquivo bruto (`calculate_sha256`) e compara com
   `EXPECTED_SHA256` (constante fixada em 12_decisoes.md).
2. Verifica contagem de linhas, nomes de colunas, tipos, nulos e strings
   vazias.
3. Se as colunas mínimas existirem, calcula: distribuição de classes,
   duplicatas exatas de `comment`, duplicatas após normalização
   (`normalize_for_comparison` — NFKC + *casefold* + colapso de espaços,
   **usada só para comparar, nunca para substituir o texto original**),
   grupos de comentários normalizados-iguais com rótulos conflitantes,
   contagem de `link_post` únicos e compartilhados entre classes, e
   presença de *rationales* por classe.
4. Cada verificação vira uma entrada booleana em `checks`; `valid` é
   `True` só se todas passarem.

**Saída:** dicionário JSON gravado em `outputs/tables/dataset_audit.json`
(por padrão) — só contém números agregados, nunca um comentário individual
ou um `link_post` individual. Isso é testado explicitamente (ver seção de
testes abaixo).

**Por que foi implementado assim:** um relatório agregado, e não uma cópia
limpa do dataset, porque o dataset tem licença CC BY-NC 4.0 e não deve ser
redistribuído mesmo indiretamente por meio de um relatório "inofensivo".

**Alternativas consideradas:** nenhuma documentada explicitamente nos
arquivos do projeto; a estrutura sugere que a alternativa óbvia (gerar um
relatório com amostras de comentários para "sanity check" visual) foi
descartada deliberadamente por risco de vazamento de dados sensíveis.

**Erros possíveis:**
- Arquivo não encontrado → `SystemExit` explícito.
- Formato não suportado (nem `.csv`, `.json`, `.jsonl`) → `ValueError` em
  `load_dataset`.
- Hash não bate → o relatório é gerado normalmente, mas
  `source_sha256_matches: false` entra em `failed_checks`, e o script sai
  com código de erro (`raise SystemExit` no fim de `main()`), mesmo assim
  grava o relatório em disco para inspeção.

**Como testar manualmente:**
```bash
python scripts/inspect_dataset.py data/raw/HateBRXplain.csv
echo $?  # 0 se válido, 1 se algum check falhar
cat outputs/tables/dataset_audit.json
```

**Testes automatizados:** `tests/test_inspect_dataset.py` — usa DataFrames
sintéticos pequenos (não o dataset real), verifica que o relatório nunca
contém o texto sensível de entrada (`"texto sensível" not in str(report)`)
e que a contagem de duplicatas normalizadas/conflitos está correta.

## 2. `src/rag_hatespeech_ptbr/splits.py` — divisão treino/validação/teste

**Objetivo:** dividir os 7.000 comentários em treino/validação/teste sem
vazamento, de forma determinística e auditável.

**Funções principais:**
- `canonicalize_group(value) -> str`: normaliza uma URL de `link_post`
  (minúsculas em esquema/host, remove barra final, descarta *query
  string*/fragmento) para que variações triviais da mesma URL não sejam
  tratadas como publicações diferentes.
- `create_grouped_manifest(frame) -> DataFrame`: núcleo do módulo.
- `build_split_report(frame, manifest, ...) -> dict`: gera o relatório
  agregado de auditoria do split.

**Entrada de `create_grouped_manifest`:** o `DataFrame` bruto completo
(precisa ter `id`, `offensive_label`, `link_post`, `comment`).

**Transformação:**
1. `_validate_input`: exige colunas presentes, sem nulos em
   `id`/`offensive_label`/`link_post`, IDs únicos, rótulos em `{0, 1}`.
2. Ordena por `id` (para determinismo independente da ordem de entrada do
   CSV — testado explicitamente).
3. Calcula `canonical_group` para cada linha.
4. Roda `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)`
   do scikit-learn, estratificando por `offensive_label` e agrupando por
   `canonical_group`. Isso garante: (a) todo `id` do mesmo grupo cai no
   mesmo fold; (b) a proporção de classes é preservada em cada fold, dentro
   do possível dado o agrupamento.
5. Fold 0 vira `test`, fold 7 vira `validation`, os outros 8 viram `train`
   (constantes `TEST_FOLD`/`VALIDATION_FOLD` no topo do arquivo).

**Saída:** um `DataFrame` de 3 colunas (`id`, `fold`, `split`), sem texto,
sem `link_post`. `write_manifest()` grava em
`data/processed/split_manifest.csv` (fora do Git) e retorna o hash
SHA-256 do arquivo escrito.

**Por que 10 folds em vez de um split direto 80/10/10:** usar
`StratifiedGroupKFold` com `n_splits=10` e escolher 1 fold para teste e 1
para validação é uma forma de obter proporções próximas de 80/10/10
**enquanto ainda se apoia numa implementação testada e citável do
scikit-learn**, em vez de escrever uma lógica de agrupamento probabilística
do zero. O trade-off, já documentado: a proporção final não é exatamente
80/10/10 (ficou 80,1%/10,1%/9,8%) porque os grupos têm tamanhos desiguais.

**Alternativas consideradas (inferidas do código e das decisões):** um
split aleatório simples por item (usado no artigo original do HateBR) foi
descartado explicitamente — 12_decisoes.md reconhece que isso torna a
divisão "não diretamente comparável à divisão por item usada no artigo
original", uma troca deliberada em favor de reduzir vazamento temático.

**Erros possíveis:**
- IDs duplicados → `ValueError("Os IDs precisam ser únicos antes da
  divisão.")`.
- Colunas obrigatórias ausentes → `ValueError` nomeando as colunas
  faltantes.
- Rótulo fora de `{0, 1}` → `ValueError`.
- (Defensivo, não deve ocorrer na prática) algum registro não receber fold
  → `RuntimeError`.

**Como testar manualmente:**
```bash
python -m scripts.create_splits data/raw/HateBRXplain.csv
cat outputs/tables/split_audit.json | python -m json.tool | less
```

O script `scripts/create_splits.py` primeiro recalcula o hash do CSV de
entrada e recusa continuar (`SystemExit`) se não bater com
`EXPECTED_SOURCE_SHA256` — ou seja, ele não deixa você gerar um split a
partir do arquivo errado por engano.

**Testes automatizados:** `tests/test_create_splits.py` — determinismo
(mesma entrada embaralhada gera o mesmo manifesto), grupos nunca
divididos entre splits, rejeição de IDs duplicados, canonicalização de URL
correta.

## 3. `src/rag_hatespeech_ptbr/data.py` — carregamento unificado (implementado nesta sessão)

**Objetivo:** ponto único para juntar o dataset bruto ao manifesto de split
e filtrar por split, para que baseline, LLM e RAG (futuros) não
reimplementem essa junção cada um à sua maneira.

**Funções principais:**
- `load_labeled_frame(dataset_path, manifest_path, verify_hashes=True) -> DataFrame`
- `get_split(frame, split_name) -> DataFrame`

**Entrada:** caminhos para o CSV bruto e o manifesto local.

**Transformação:** verifica os dois hashes contra as constantes já
existentes em `splits.py` (reaproveitadas, não duplicadas), faz
`merge(..., how="inner", validate="one_to_one")` por `id`, e levanta
`DatasetIntegrityError` se algum `id` do dataset não tiver entrada no
manifesto (o que indicaria manifesto desatualizado).

**Saída:** um `DataFrame` com todas as colunas do dataset bruto mais
`fold`/`split`.

**Por que foi implementado assim:** a verificação de hash **por padrão
ligada** (`verify_hashes=True`) é uma decisão deliberada — o custo de
esquecer de verificar manualmente antes de treinar um modelo é alto
(treinar sobre um split não aprovado, silenciosamente). `verify_hashes=False`
existe só para testes com dados sintéticos, que nunca vão bater com os
hashes reais de qualquer forma.

**Alternativa considerada e descartada:** duplicar a lógica de
carregamento dentro de `baseline.py` ou de `run_baseline.py`. Rejeitada
porque o mesmo carregamento será necessário para o baseline de LLM e para o
RAG — centralizar aqui evita reescrever (e possivelmente divergir) a lógica
de verificação de integridade em três lugares.

**Erros possíveis:**
- Arquivo de dataset ou manifesto não encontrado → `FileNotFoundError`.
- Hash divergente → `DatasetIntegrityError`.
- `id` do dataset sem entrada no manifesto → `DatasetIntegrityError`.
- `get_split` com nome de split desconhecido → `ValueError`.

**Como testar manualmente:**
```python
from pathlib import Path
from rag_hatespeech_ptbr.data import load_labeled_frame, get_split

frame = load_labeled_frame(Path("data/raw/HateBRXplain.csv"), Path("data/processed/split_manifest.csv"))
train = get_split(frame, "train")
print(len(train))  # 5608
```

**Testes automatizados:** `tests/test_data.py`.

## 4. `src/rag_hatespeech_ptbr/metrics.py` — métricas compartilhadas (implementado nesta sessão)

**Objetivo:** um único lugar que calcula F1/precisão/revocação da classe
ofensiva, macro-F1 e matriz de confusão — reaproveitável por qualquer
modelo futuro, incluindo a análise por estrato de sarcasmo.

**Função principal:** `evaluate_predictions(y_true, y_pred, mask=None) -> ClassificationMetrics`.

**Entrada:** duas sequências de rótulos (0/1) do mesmo tamanho, e um
`mask` booleano opcional.

**Transformação:** se `mask` for passado, filtra `y_true`/`y_pred` antes de
calcular qualquer métrica (isso é o que vai permitir, no futuro, calcular
métricas só para os itens marcados como sarcásticos, sem duplicar código).
Usa `sklearn.metrics.confusion_matrix`/`f1_score`/`precision_score`/
`recall_score` com `labels=[0, 1]` fixo, para garantir uma matriz 2×2
mesmo se uma classe não aparecer nas previsões.

**Saída:** um `ClassificationMetrics` (dataclass congelada), com
`.to_dict()` para serializar em JSON.

**Por que foi implementado assim:** dataclass em vez de dicionário solto,
para que erros de nome de chave (`"f1"` vs `"positive_f1"`) virem erro de
atributo em tempo de desenvolvimento, não um `KeyError` silencioso em
produção de resultado.

**Alternativas consideradas:** usar `sklearn.metrics.classification_report`
diretamente. Rejeitada porque seu formato de saída (string ou dict
aninhado por classe) é mais difícil de padronizar entre baseline/LLM/RAG do
que um objeto com campos fixos e nomeados.

**Erros possíveis:**
- Tamanhos diferentes de `y_true`/`y_pred` → `ValueError`.
- Lista vazia → `ValueError` (evita um F1 sem sentido sobre zero amostras).
- `mask` de tamanho errado → `ValueError`.

**Como testar manualmente:**
```python
from rag_hatespeech_ptbr.metrics import evaluate_predictions
m = evaluate_predictions([0, 1, 0, 1], [0, 1, 1, 1])
print(m.positive_f1, m.confusion_matrix)
```

**Testes automatizados:** `tests/test_metrics.py` — casos de acerto
perfeito, erro total, aplicação de `mask`, e as três validações de erro.

## 5. `src/rag_hatespeech_ptbr/baseline.py` — baseline determinístico (implementado nesta sessão)

**Objetivo:** primeiro modelo real do projeto — responde ao objetivo
específico do PCC "analisar as limitações de classificadores textuais
diante de comentários curtos, sarcásticos e ambíguos" com um número
concreto, sem depender de LLM.

**Funções principais:**
- `build_pipeline(random_state=42) -> sklearn.pipeline.Pipeline`
- `train_baseline(train_frame, random_state=42) -> Pipeline`
- `evaluate_baseline(pipeline, frame) -> ClassificationMetrics`

**Entrada de `train_baseline`:** um `DataFrame` já filtrado para
`split == "train"` (via `data.get_split`).

**Transformação:** `Pipeline` do scikit-learn com dois passos:
1. `TfidfVectorizer(lowercase=True, strip_accents=None, ngram_range=(1,2),
   min_df=2, sublinear_tf=True)` — uni+bigramas de palavras, acentos
   preservados (decisão deliberada: acentuação carrega sinal em PT-BR),
   `min_df=2` para não deixar o vocabulário explodir com termos que
   aparecem uma única vez.
2. `LogisticRegression(max_iter=1000, class_weight="balanced",
   random_state=42)` — `class_weight="balanced"` é defensivo (as classes já
   são exatamente balanceadas por construção do split, então não deveria
   mudar muito o resultado, mas protege contra qualquer desbalanceamento
   inesperado).

`pipeline.fit(X, y)` ajusta os dois passos em sequência — o vocabulário do
TF-IDF **e** os coeficientes da regressão só veem `train_frame`.

**Saída de `train_baseline`:** o `Pipeline` já ajustado. `evaluate_baseline`
chama `pipeline.predict()` (que internamente só faz `.transform()` no
TF-IDF, nunca `.fit()`) sobre qualquer outro `DataFrame`, e delega o
cálculo de métricas para `metrics.evaluate_predictions`.

**Por que foi implementado assim:** usar `sklearn.pipeline.Pipeline` em vez
de chamar `TfidfVectorizer` e `LogisticRegression` separadamente é o que
torna a garantia "vocabulário aprendido só no treino" estrutural — não
depende de o programador lembrar de chamar `.fit()` e `.transform()` na
ordem certa toda vez.

**Alternativas consideradas:**
- Um classificador "burro" (maioria da classe) como piso ainda mais básico
  — não implementado nesta sessão porque as classes são balanceadas por
  construção (50/50 em todos os splits), então esse baseline teria F1 = 0
  na classe positiva de forma trivial e pouco informativa. Pode valer a
  pena adicionar como uma linha extra na tabela final do TCC, mas é
  opcional (ver 5_decisoes_pendentes.md para não decidir isso
  silenciosamente).
- Um modelo baseado em contagem simples de palavras ofensivas
  (*blocklist*) — descartado porque o próprio PCC (1.2) já argumenta que
  esse tipo de abordagem falha exatamente nos casos que o projeto quer
  estudar (ódio implícito/sarcástico).

**Erros possíveis:** nenhum tratamento de erro customizado — se
`train_frame` estiver vazio ou só tiver uma classe, o scikit-learn levanta
seus próprios erros (`ValueError` do `LogisticRegression`). Não foi
adicionado tratamento extra porque `data.get_split` já garante que o split
"train" existe e tem ambas as classes, dado que o manifesto foi validado
(7_arquitetura_pipeline.md, princípio de só validar em fronteiras).

**Como testar manualmente:**
```python
from rag_hatespeech_ptbr.data import load_labeled_frame, get_split
from rag_hatespeech_ptbr.baseline import train_baseline, evaluate_baseline
from pathlib import Path

frame = load_labeled_frame(Path("data/raw/HateBRXplain.csv"), Path("data/processed/split_manifest.csv"))
pipeline = train_baseline(get_split(frame, "train"))
metrics = evaluate_baseline(pipeline, get_split(frame, "validation"))
print(metrics.to_dict())
```

**Testes automatizados:** `tests/test_baseline.py` — verifica que o
vocabulário aprendido não contém palavras exclusivas de um exemplo de
validação sintético (evidência indireta de que `.fit()` só viu o treino),
que duas classes trivialmente separáveis são de fato separadas (F1 = 1.0
em um cenário sintético fácil), e que duas execuções com a mesma semente
produzem previsões idênticas.

## 6. `scripts/run_baseline.py` — CLI de execução e registro (implementado nesta sessão)

**Objetivo:** ponto de entrada reprodutível por linha de comando, que
treina, avalia, e grava um registro de experimento — sem exigir que quem
for rodar o pipeline escreva código Python.

**Fluxo de `main()`:**
1. Se `--split test` for pedido sem `--confirm-final-test-run`, sai
   imediatamente com `SystemExit` e uma mensagem explicando o protocolo de
   execução única no teste.
2. Carrega e junta dados via `data.load_labeled_frame` (hashes verificados
   por padrão).
3. Treina em `train`, avalia no split pedido (`validation` por padrão).
4. Monta um dicionário `record` com: ID do experimento (nome do modelo +
   sufixo aleatório curto), timestamp UTC, hash do commit Git atual (via
   `git rev-parse HEAD`, tolerante a falha — retorna `None` se não estiver
   num repositório Git ou o comando falhar), hiperparâmetros, split
   avaliado, tamanhos de treino/avaliação, hashes do dataset e do
   manifesto usados (mais os hashes esperados, para comparação visual no
   próprio JSON), e as métricas.
5. Grava esse `record` duas vezes: em `outputs/experiments/<id>.json`
   (bruto, não versionado — ver `.gitignore`) e em
   `outputs/tables/baseline_<split>.json` (canônico, versionado,
   sobrescrito a cada execução — mesmo padrão de
   `dataset_audit.json`/`split_audit.json`).
6. Se `--split test`, também acrescenta uma linha a
   `outputs/experiments/test_evaluations_ledger.jsonl` (append, nunca
   sobrescrito) e imprime um aviso.

**Por que dois arquivos de saída (bruto + canônico):** o bruto preserva o
histórico de todas as execuções (útil para depuração, mas não deve poluir
o Git); o canônico é o "resultado atual" que o TCC deve citar, seguindo
exatamente o padrão já estabelecido por `inspect_dataset.py` e
`create_splits.py`.

**Erros possíveis:**
- Dataset ou manifesto ausente/hash divergente → propagado de
  `load_labeled_frame` (`FileNotFoundError`/`DatasetIntegrityError`).
- `git rev-parse` falhar (não é um repositório Git, ou Git não instalado)
  → capturado, `git_commit` fica `None`, a execução continua normalmente.

**Como executar exatamente:**
```bash
# avaliação padrão (validação) — segura para rodar quantas vezes quiser
python -m scripts.run_baseline data/raw/HateBRXplain.csv

# avaliação no teste — só quando o pipeline completo estiver pronto
python -m scripts.run_baseline data/raw/HateBRXplain.csv --split test --confirm-final-test-run
```

**Testes automatizados:** nenhum teste de CLI dedicado (mesmo padrão já
usado por `scripts/create_splits.py`, que também não tem teste de CLI
próprio — a lógica testável está toda nas funções de `data.py`,
`baseline.py` e `metrics.py`, que o script apenas orquestra).

## 7. O que ainda não existe (e por quê)

As etapas abaixo não têm código no repositório. Cada uma está descrita em
detalhe de design (não de implementação) em 7_arquitetura_pipeline.md:

- **Recuperação/indexação (RAG)** — provedor de embeddings e banco vetorial
  já decididos (Gemini, tentando Pinecone primeiro — 12_decisoes.md), falta
  implementar.
- **Prompts** — depende da implementação do cliente Gemini.
- **Execução de modelo (com e sem RAG)** — nenhuma chamada de API foi feita
  ainda; falta implementar o cliente (com modo *dry-run* antes do real).
- **Avaliação final no teste** — depende de tudo acima estar pronto, para
  respeitar a execução única prevista no protocolo.
- **Anotação e análise por estrato de sarcasmo** — depende de anotação
  humana (piloto + teste), que ainda não começou.

Quando esses componentes forem implementados, este documento deve ganhar
seções novas seguindo exatamente o mesmo formato usado acima (objetivo,
arquivo, função principal, entrada, transformação, saída, por que assim,
alternativas, erros possíveis, como testar, comando exato).
