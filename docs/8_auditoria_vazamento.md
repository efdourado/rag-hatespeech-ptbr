# Auditoria de vazamento (leakage)

Este documento lista, um a um, os mecanismos de vazamento relevantes para
este projeto, o que previne cada um, e a evidência concreta no código ou nos
relatórios agregados. Escopo: tudo que existe hoje (auditoria, split,
baseline). Os mecanismos relativos a RAG/LLM/anotação de sarcasmo são
descritos como **risco futuro** com o desenho de prevenção planejado, já que
o código correspondente ainda não existe.

## 1. Duplicatas entre splits

**Risco:** o mesmo comentário (ou uma variação trivial dele) aparecer em
mais de um split, permitindo que um modelo "memorize" no treino algo que
será perguntado na validação/teste.

**Prevenção:**
- Checagem de duplicatas exatas de texto (`comment.duplicated()`).
- Checagem de duplicatas após normalização (Unicode NFKC, *casefold*, espaços
  colapsados) — pega variações de caixa/espaçamento que a checagem exata não
  pegaria.
- Checagem de que grupos de duplicatas não têm rótulos conflitantes.
- Todas as três feitas **antes** do split, sobre o dataset inteiro
  (`scripts/inspect_dataset.py`), e **depois** do split, por par de splits
  (`src/rag_hatespeech_ptbr/splits.py::_pairwise_overlap` aplicado a
  `normalized_comment`).

**Evidência:**
- `outputs/tables/dataset_audit.json`:
  `exact_duplicate_comments: 0`, `normalized_duplicate_comments: 0`,
  `conflicting_normalized_label_groups: 0`.
- `outputs/tables/split_audit.json`:
  `pairwise_normalized_comment_overlap` = 0 para os três pares de splits;
  check `normalized_comments_do_not_cross_splits: true`.
- Testado em `tests/test_create_splits.py::test_split_report_contains_only_aggregate_information`.

**Status:** coberto, verificado nesta sessão (reexecução byte-idêntica).

## 2. Vazamento por agrupamento (mesma publicação em splits diferentes)

**Risco:** comentários de uma mesma publicação (`link_post`) tendem a
compartilhar vocabulário, alvo e contexto. Se comentários da mesma
publicação caírem em splits diferentes, um modelo pode aprender padrões
específicos daquela publicação no treino e "acertar fácil" no
teste — não porque generalizou, mas porque viu contexto da mesma thread.

**Prevenção:** `StratifiedGroupKFold` agrupado por `link_post` canônico
(normalizado por `canonicalize_group()`, que ignora variação de
maiúsculas/minúsculas, barra final e *query string*/fragmento de URL).
Garante que todo o grupo cai inteiro em um único split.

**Evidência:**
- `outputs/tables/split_audit.json`: `pairwise_group_overlap` = 0 para os
  três pares; check `groups_do_not_cross_splits: true`.
- Testado em `tests/test_create_splits.py::test_grouped_manifest_is_deterministic_and_keeps_groups_together`.

**Limitação documentada, não um vazamento, mas correlacionada:** a
concentração de itens por publicação dentro de cada split é alta (maior
grupo = 45,9% do teste). Isso não é vazamento entre splits, mas reduz a
independência estatística das observações dentro de um split — por isso
`12_decisoes.md` (2026-08-04) exige reamostragem por `link_post` nos
intervalos de confiança. Essa reamostragem **ainda não está implementada em
código** — listada como trabalho pendente em `7_arquitetura_pipeline.md`
(item 12, ao ser estendido).

**Status:** coberto para o requisito de não-sobreposição; a mitigação
estatística da concentração é uma decisão registrada, mas não codificada
ainda.

## 3. Retrieval usando validação/teste

**Risco:** o índice RAG incluir, direta ou indiretamente, exemplos de
validação ou teste — o que tornaria a tarefa artificialmente fácil (o
modelo recuperaria o próprio exemplo, ou algo quase idêntico a ele).

**Prevenção planejada (nenhum código de RAG existe ainda):**
- `12_decisoes.md` (2026-08-04): "Política RAG: somente IDs marcados
  como `train` podem ser indexados ou recuperados como contexto."
- `7_arquitetura_pipeline.md`, item 8: o índice deve ser construído
  chamando `get_split(frame, "train")` (já implementado em
  `src/rag_hatespeech_ptbr/data.py`) antes de qualquer *embedding* — nunca
  sobre o `frame` completo.

**Verificação recomendada quando o índice existir:** um teste automatizado
que constrói o índice a partir de um `DataFrame` sintético com IDs
conhecidos de todos os splits, e afirma que nenhum ID de `validation`/`test`
aparece nos vetores indexados — no mesmo espírito de
`tests/test_create_splits.py`.

**Status:** **risco futuro, ainda não aplicável** — não há código de
retrieval para auditar. Este item deve ser revisitado assim que
`7_arquitetura_pipeline.md` item 8 for implementado.

## 4. Ajuste de prompt/hiperparâmetro pelo resultado do teste

**Risco:** iterar em prompt, `top-k`, ou hiperparâmetros do LLM observando o
desempenho no teste, o que transforma o teste em um conjunto de validação
disfarçado e infla os números reportados.

**Prevenção:**
- `9_protocolo_pesquisa.md`: "Não ajustar prompt ou hiperparâmetros pelo
  resultado final do teste."
- Mecanismo técnico implementado nesta sessão: `scripts/run_baseline.py`
  recusa avaliar no split de teste sem `--confirm-final-test-run`, e cada
  uso é anexado a `outputs/experiments/test_evaluations_ledger.jsonl` — um
  registro cumulativo e append-only, então uma segunda "olhada" no teste
  fica visível no histórico do arquivo, mesmo que involuntária.
- **Limitação honesta:** este mecanismo é uma barreira de fricção e
  auditoria, não uma barreira criptográfica. Ele impede o esquecimento
  acidental, mas não impede alguém decidido a burlar o protocolo. A
  disciplina real depende do autor seguir a política registrada em
  `12_decisoes.md` (2026-08-15).
- **Ainda não usado:** o baseline determinístico foi avaliado apenas na
  validação nesta sessão; o teste permanece intocado
  (`outputs/experiments/test_evaluations_ledger.jsonl` não existe ainda no
  repositório — sua ausência é, ela mesma, evidência de que o teste nunca
  foi avaliado até agora).

**Status:** mecanismo técnico implementado; disciplina de uso depende do
autor ao longo do restante do TCC.

## 5. Anotação contaminada pela previsão do modelo

**Risco:** o anotador humano de sarcasmo ver a saída do classificador (ou a
sugestão de pré-anotação por IA) antes de decidir, enviesando a anotação
para concordar com o modelo.

**Prevenção documentada (a anotação ainda não ocorreu):**
- `11_diretrizes_anotacao.md`: "Anotar sem consultar a sugestão da IA."
  E na seção "Não fazer": "Não alterar o rótulo por conhecer a previsão do
  sistema avaliado."
- Estrutura de campos já prevê registrar `ai_sarcasm_label` e
  `ai_confidence` **separadamente** de `human_sarcasm_label`, o que permite
  auditar depois se houve concordância suspeita.

**Status:** protocolo definido; **não verificável ainda**, porque nenhuma
anotação foi feita. Quando o piloto começar, recomenda-se registrar também
a ordem/timing da anotação (decisão humana registrada antes de revelar a
sugestão da IA), não apenas o valor final, para permitir auditoria
posterior.

## 6. Pré-processamento aprendido sobre o conjunto completo

**Risco:** ajustar (`fit`) qualquer transformação estatística — vetorizador,
normalizador, seleção de features, escala — usando dados de validação ou
teste, mesmo que indiretamente (ex.: calcular `min_df` do TF-IDF olhando o
dataset inteiro antes de decidir o valor).

**Prevenção:**
- `src/rag_hatespeech_ptbr/baseline.py::train_baseline` chama
  `pipeline.fit()` **apenas** com `train_frame`. O `Pipeline` do
  scikit-learn garante que `TfidfVectorizer.fit()` (que define o
  vocabulário) só é chamado nesse `.fit()` — `evaluate_baseline` chama
  exclusivamente `.predict()`, que internamente usa `.transform()`, nunca
  `.fit()`.
- Os hiperparâmetros do TF-IDF (`ngram_range`, `min_df`, `sublinear_tf`)
  foram escolhidos por valores padrão razoáveis da literatura, não
  calibrados observando desempenho em validação ou teste — não houve
  nenhuma busca de hiperparâmetros nesta sessão.

**Evidência automatizada:**
`tests/test_baseline.py::test_train_baseline_fits_vectorizer_only_on_train_frame`
verifica que uma palavra presente apenas em um exemplo de validação
hipotético não aparece no vocabulário aprendido do treino.

**Status:** coberto para o baseline atual. Quando um modelo de embeddings
for escolhido (`5_decisoes_pendentes.md`), a mesma disciplina deve se
aplicar: qualquer normalização aprendida (ex.: escala de vetores) deve ser
ajustada só no treino.

## 7. Vazamento indireto via `link_post` compartilhado entre classes

**Risco:** 53 dos 85 `link_post` têm comentários das duas classes
(ofensivo e não ofensivo). Isso não é vazamento entre splits (já resolvido
pelo agrupamento), mas é um lembrete de que o **contexto da publicação**
(ex.: quem é o político, o teor do post original) pode ser um atalho que
não generaliza — um risco de validade, não de vazamento técnico.

**Prevenção:** não aplicável a vazamento; é uma limitação a ser discutida na
seção de limitações do TCC (ver `13_guia_escrita_tcc.md`), não um defeito a
corrigir no pipeline.

**Status:** documentado como limitação conhecida, não como falha.

## 8. Vazamento futuro específico do RAG: *rationale* revelando o rótulo

**Risco:** ao montar o prompt de inferência para um item de
validação/teste, incluir sem querer o *rationale* do **próprio item que
está sendo classificado** (em vez de apenas *rationales* de exemplos
recuperados do treino), o que equivaleria a dar a resposta ao modelo.

**Prevenção planejada:** o prompt deve ser montado a partir de dois objetos
distintos e nunca confundidos: (a) o comentário-alvo, sem rótulo nem
*rationale*; (b) os `top_k` exemplos recuperados do índice (treino), cada um
com seu próprio `comment` + `rationale`. Como o índice em si já é restrito
ao treino (item 3 acima), este risco só existiria se o código de montagem do
prompt buscasse por engano o *rationale* do item de validação/teste em vez
de ignorá-lo. Recomenda-se, quando esse código existir, um teste que
verifica que a *string* do prompt nunca contém o `id` ou o `rationale` do
item avaliado.

**Status:** risco futuro, nenhum código existe ainda para auditar.

## Resumo

| # | Mecanismo | Status atual |
|---|---|---|
| 1 | Duplicatas entre splits | Coberto e verificado |
| 2 | Vazamento por agrupamento (`link_post`) | Coberto e verificado; mitigação estatística da concentração ainda não codificada |
| 3 | Retrieval usando validação/teste | Risco futuro — sem código de RAG ainda |
| 4 | Ajuste de prompt/hiperparâmetro pelo teste | Mecanismo técnico implementado; disciplina de uso é responsabilidade contínua |
| 5 | Anotação contaminada por previsão do modelo | Protocolo definido; não verificável até a anotação ocorrer |
| 6 | Pré-processamento aprendido no conjunto completo | Coberto para o baseline; regra a repetir em embeddings futuros |
| 7 | `link_post` compartilhado entre classes | Limitação de validade, não vazamento — documentar, não corrigir |
| 8 | *Rationale* do próprio item vazando no prompt do RAG | Risco futuro — sem código de RAG ainda |
