# Guia de escrita do TCC

Este arquivo reúne dois instrumentos que devem ser usados juntos ao
escrever o TCC: uma matriz de rastreabilidade (o que o PCC promete vs. o
que o experimento efetivamente testa) e uma lista de critérios
verificáveis (como saber se cada parte do TCC está pronta). É o último
arquivo numerado de `docs/` de propósito — o objetivo é que ele seja o
guia principal na hora de sentar para escrever o texto.

## Parte 1 — Rastreabilidade PCC → TCC

Objetivo: garantir que nada no texto do TCC prometa algo que o experimento
não testa, e que nada no experimento faça algo sem justificativa no texto.
Fonte primária do lado "texto": `2-pcc-final.pdf` (versão final, 22/07/2026).
Fonte do lado "experimento": código e artefatos em `docs/`, `scripts/`,
`src/`, `outputs/` neste repositório.

Convenção de status:
- **Feito** — existe evidência executável (código + saída) para a etapa.
- **Planejado** — existe decisão/desenho documentado, sem código ainda.
- **Bloqueado** — não pode avançar sem decisão externa ou sem uma etapa
  humana (anotação).
- **Não iniciado** — nada existe ainda, sem bloqueio conhecido além de
  ordem de execução.

### Matriz principal

| Problema (PCC 1.2) | Objetivo geral (PCC 1.5.1) | Objetivo específico (PCC 1.5.2) | Pergunta/hipótese | Dataset | Método | Experimento | Métrica | Seção do TCC | Status |
|---|---|---|---|---|---|---|---|---|---|
| Classificadores isolados falham em ódio implícito/sarcástico por falta de contexto | Desenvolver e avaliar abordagem RAG+LLM para apoiar detecção de sarcasmo associado a ódio | "Revisar a literatura sobre detecção de sarcasmo, discurso de ódio e uso de contexto" | — | — | Revisão estruturada (não sistemática) | Revisão em IEEE Xplore + ACM DL + Google Acadêmico, com critérios de inclusão/exclusão | — | Cap. 3 do PCC (já escrito) | **Feito** (na versão do PCC; TCC deve apenas manter/atualizar se necessário) |
| idem | idem | "Analisar as limitações de classificadores textuais diante de comentários curtos, sarcásticos e ambíguos" | Ligado à pergunta principal (baseline vs RAG) | HateBRXplain, conjunto de validação/teste | Baseline determinístico (TF-IDF + regressão logística) como piso de referência | `scripts/run_baseline.py` | F1 da classe ofensiva, precisão, revocação, matriz de confusão | Metodologia/Resultados do TCC (a escrever) | **Feito** o código e a execução na validação; **não iniciado** o texto |
| idem | idem | "Projetar uma abordagem baseada em RAG para recuperar exemplos anotados e justificativas do dataset HateBRXplain" | idem | Índice restrito ao `split == "train"` | Recuperação por similaridade de cosseno sobre *embeddings* (Gemini) + *rationales* como contexto | Nenhum código ainda | — | Cap. 5 do TCC | **Planejado**: provedor e estratégia de indexação já decididos (12_decisoes.md), falta implementar |
| idem | idem | "Avaliar o desempenho da abordagem proposta em comparação com configurações sem recuperação contextual" | Pergunta principal do protocolo | Teste (avaliação única) | Comparar Gemini sem RAG vs. Gemini com RAG, mesmo modelo, mesmo prompt base | Nenhum código ainda | F1 principal + precisão/revocação/macro-F1/matriz de confusão | Cap. 5 e 6 do TCC | **Planejado**: provedor decidido, falta implementar |
| idem | idem | "Analisar a contribuição das justificativas recuperadas para a explicabilidade das classificações" | Direção de desenho aceita (formulário de avaliação por terceiros) | Saída qualitativa do Gemini (classe + justificativa) | Formulário/interface coletando opiniões de um grupo de pessoas sobre a qualidade das justificativas, comparadas ao *rationale* humano recuperado | Nenhum código ainda | Sem métrica quantitativa fechada — depende do desenho do formulário | Cap. 5/6 do TCC | **Planejado**: direção aceita (12_decisoes.md), detalhes pendentes (5_decisoes_pendentes.md, D8) |
| Sarcasmo mascara ódio e não é rótulo do dataset | idem (o PCC já trata sarcasmo como estrato, não como classe) | Implícito no objetivo geral | "melhora a classificação... especialmente em comentários sarcásticos" | Amostra do teste anotada por humano | `human_sarcasm_label` como variável de estrato, nunca como alvo de treino | 11_diretrizes_anotacao.md (protocolo pronto, execução pendente; segundo anotador confirmado) | F1 por estrato sarcástico/não sarcástico | Cap. 5/6 do TCC | **Bloqueado**: tarefa humana não delegável |

### Pontos de atenção específicos

**1. "Ofensivo / Sarcasmo" como classe única (risco já corrigido no PCC).**
A Tabela 5.1 da versão final do PCC (pág. 27) mostra apenas `Ofensivo` /
`Não ofensivo` como classes, com nota explicando que sarcasmo não tem
rótulo próprio. Confirmado corrigido. Nenhuma tabela gerada até agora
(`outputs/tables/*.json`) usa sarcasmo como rótulo nativo.

**2. Tecnologias nomeadas no PCC vs. decisões posteriores.** O PCC (Cap.
4) nomeia especificamente: Python + FastAPI, LangChain, Pinecone,
`text-embedding-3-small` ou `multilingual-e5`, GPT-4o-mini ou Llama 3.
Decisão atual (12_decisoes.md, 2026-08-15): Gemini como provedor de
LLM/embeddings (diferente do PCC, que não cita Gemini); Pinecone e
LangChain serão tentados primeiro, na mesma linha do PCC, com
*fallback* local documentado se necessário; FastAPI incluído no escopo.
**Ação para o texto do TCC:** o capítulo de metodologia deve declarar
explicitamente que o provedor de LLM final foi o Gemini (não os
candidatos citados no PCC), e explicar o motivo (acesso via assinatura do
autor) — isso é uma mudança real em relação ao PCC e precisa aparecer
como tal, não ser silenciada.

**3. Agrupamento por `link_post` no split (metodologia além do PCC).** O
PCC não menciona agrupar o split por publicação de origem. Decisão tomada
depois (12_decisoes.md, 2026-08-04), motivada por um achado concreto da
auditoria: 53 dos 85 `link_post` aparecem nas duas classes, o que criaria
vazamento temático se comentários da mesma publicação fossem divididos
entre treino e teste. É uma melhoria metodológica não descrita no PCC, não
um desvio do tema — o PCC (5.2) já deixava em aberto "a proporção dos
conjuntos e os critérios definitivos de divisão serão apresentados no
TCC".

**4. Métricas além do que o PCC promete.** O PCC (5.4) promete matriz de
confusão, F1 (principal), precisão, revocação. 9_protocolo_pesquisa.md
acrescenta macro-F1, Cohen's kappa e reamostragem por `link_post` para
intervalos de confiança — motivada pela concentração de comentários por
publicação encontrada na auditoria (até 45,9% do teste em um único
`link_post`). Extensão, não contradição — apresentar como refinamento
motivado pelos dados reais.

**5. Cronograma (PCC 5.5) vs. andamento real.** O cronograma do PCC situa
"Implementação: Ingestão de Dados" em TCC-M1 e "Implementação: API e
LangChain" em TCC-M2/M3. O projeto está adiantado na parte de dados
(auditoria, split e baseline já prontos) e a anotação de sarcasmo — que
deveria estar próxima do início do semestre de TCC — ainda não começou.
Estado de fato a monitorar, não uma inconsistência documental.

### Itens sem contrapartida experimental ainda

- **Explicabilidade das justificativas** (objetivo específico 5): a
  direção do desenho já foi aceita (formulário de avaliação), mas os
  detalhes (perguntas, amostra, participantes) só podem ser fechados
  quando o RAG estiver produzindo saídas reais.
- **Ablação sem *rationales*** (9_protocolo_pesquisa.md, item 9, "se
  prazo e custo permitirem"): condicional, não uma promessa firme — se
  entrar no texto final como resultado, precisa ter sido de fato
  executada.

### Itens do experimento sem justificativa textual

Nenhum encontrado até o momento. O baseline determinístico corresponde
exatamente ao item 1 de "O que vem depois" em 10_plano_dados.md, que por
sua vez decorre do objetivo específico 2 do PCC.

---

## Parte 2 — Critérios verificáveis para o TCC pronto

Cada critério pode ser checado olhando um artefato concreto — código,
arquivo, número — não por impressão subjetiva.

### 1. Coerência científica

- [ ] Todo objetivo específico do PCC (1.5.2) corresponde a pelo menos uma
  etapa metodológica em 9_protocolo_pesquisa.md **e** a um resultado ou
  evidência em `outputs/`. Verificável na matriz acima — nenhuma linha
  pode ficar em "Não iniciado" no momento da entrega.
- [ ] A pergunta de pesquisa declarada no TCC é idêntica, palavra por
  palavra ou em paráfrase estritamente equivalente, à de
  9_protocolo_pesquisa.md. Nenhuma pergunta nova pode aparecer no texto
  final sem uma entrada correspondente em 12_decisoes.md explicando
  quando e por que mudou.
- [ ] Existe uma hipótese quantitativa explícita registrada antes da
  avaliação final no teste. Uma proposta de referência já existe
  (12_decisoes.md, ~6–7 pontos percentuais de diferença mínima
  detectável) — falta confirmar com o orientador e recalcular com dados
  reais (5_decisoes_pendentes.md, D7).
- [ ] Nenhuma alegação no texto atribui capacidade de "detectar sarcasmo"
  ao sistema como um todo — o sistema classifica ofensividade; sarcasmo é
  estrato de análise.

### 2. Texto

- [ ] Título, resumo, objetivo geral e conclusão usam a mesma formulação
  do problema (não alternam entre "detectar sarcasmo", "detectar ódio" e
  "classificar ofensividade em casos sarcásticos" como se fossem
  sinônimos — ponto já corrigido entre PCC rascunho e PCC final).
- [ ] Toda citação de um número (F1, precisão, contagem de itens, hash) no
  texto corrido tem uma fonte rastreável em `outputs/tables/*.json` ou
  `outputs/experiments/*.json`.
- [ ] Toda menção a uma decisão metodológica tem uma entrada
  correspondente em 12_decisoes.md, com data e motivo.
- [ ] Nenhuma seção do texto final descreve como "provisório" algo que já
  está congelado (ex.: a pergunta de pesquisa, o desenho do split).

### 3. Método

- [ ] O capítulo de metodologia declara explicitamente que o Gemini foi o
  provedor de LLM/embeddings usado (diferente dos candidatos citados no
  PCC), e se Pinecone/LangChain/FastAPI foram usados como no PCC ou
  substituídos — sem silêncio sobre a divergência.
- [ ] O capítulo de metodologia explica o agrupamento por `link_post` e a
  reamostragem por publicação nos intervalos de confiança.
- [ ] O tratamento do rótulo `incerto` de sarcasmo na análise principal
  está definido e é o mesmo em todo o texto.
- [ ] Está explícito no texto que os *rationales* justificam ofensividade,
  não sarcasmo.

### 4. Código

- [ ] `python -m pytest` passa 100% no momento da entrega.
- [ ] `ruff check .` não reporta nenhum erro.
- [ ] Nenhum caminho absoluto específico de uma máquina aparece em
  `scripts/`, `src/` ou `tests/` — verificável com `grep -rn "/home/"
  scripts/ src/ tests/` retornando vazio.
- [ ] Nenhuma credencial, chave de API ou *token* aparece em nenhum
  arquivo versionado — importante a partir de agora, com o cliente
  Gemini/Pinecone entrando em cena.
- [ ] Toda função pública nova (LLM, RAG, análise por sarcasmo) tem pelo
  menos um teste que falha se a lógica de vazamento for quebrada.
- [ ] Todo script em `scripts/` é executável isoladamente com um comando
  documentado no README ou em 6_guia_codigo.md.

### 5. Dataset

- [ ] Nenhuma tabela do TCC apresenta sarcasmo como rótulo nativo do
  HateBRXplain — toda coluna de sarcasmo deve indicar que é anotação
  própria do projeto (`human_sarcasm_label`).
- [ ] `data/raw/HateBRXplain.csv` nunca é versionado no Git.
- [ ] O hash SHA-256 do dataset usado em cada resultado bate com
  `EXPECTED_SOURCE_SHA256` em `src/rag_hatespeech_ptbr/splits.py`.
- [ ] O manifesto de split usado para gerar os resultados finais bate com
  `EXPECTED_MANIFEST_SHA256`.

### 6. Resultados

- [ ] Todo resultado quantitativo é reproduzível executando um comando
  documentado — não existe número no texto sem comando correspondente.
- [ ] O conjunto de teste foi avaliado uma única vez por combinação de
  modelo final reportada no TCC. Verificável em
  `outputs/experiments/test_evaluations_ledger.jsonl`.
- [ ] A tabela comparativa final (baseline vs. Gemini sem RAG vs. RAG) usa
  a mesma partição de teste para todos os modelos.
- [ ] O F1 da classe ofensiva é sempre a métrica principal, com
  precisão/revocação/macro-F1/matriz de confusão como complementares, na
  mesma ordem em toda tabela.

### 7. Estatística

- [ ] Toda comparação entre modelos vem acompanhada de intervalo de
  confiança ou teste estatístico, reamostrado por `link_post`.
- [ ] A análise de sensibilidade sem o maior `link_post` do teste está de
  fato presente nos resultados, não só mencionada como planejada.
- [ ] Se Cohen's kappa for reportado, o método de cálculo e o tamanho da
  amostra dupla-anotada estão declarados.

### 8. Figuras

- [ ] Toda figura tem fonte rastreável (script + comando).
- [ ] Nenhuma figura usa sarcasmo como se fosse classe do dataset
  original.
- [ ] Figuras comparativas usam a mesma escala de eixo Y entre si.

### 9. Referências

- [ ] Toda afirmação bibliográfica relevante foi conferida contra a fonte
  primária, não contra um resumo de segunda mão.
- [ ] Toda citação no texto tem entrada correspondente nas Referências, e
  vice-versa.
- [ ] Nomes de modelos/ferramentas de terceiros citados no texto final
  correspondem ao que foi de fato usado — não ao que foi só cogitado no
  PCC (ex.: Gemini, não GPT-4o-mini/Llama 3, se essa for a escolha final).

### 10. Reprodutibilidade

- [ ] Um terceiro com o dataset oficial e o repositório consegue
  reproduzir `dataset_audit.json`, `split_audit.json` e
  `baseline_validation.json` byte a byte, seguindo só o README (já
  verificado para os dois primeiros; reverificar sempre que `baseline.py`
  mudar).
- [ ] Toda dependência de terceiro (embeddings, LLM) usada nos resultados
  finais tem versão/modelo fixado e registrado no experimento.
- [ ] `pyproject.toml` fixa as versões usadas para gerar os resultados
  finais (scikit-learn já fixado; pandas a fixar na rodada final —
  12_decisoes.md).

### 11. Limitações

- [ ] A concentração de comentários por `link_post` está descrita
  explicitamente como limitação de validade estatística.
- [ ] A ausência de rótulo de sarcasmo nativo do dataset, e a dependência
  de anotação própria, está descrita como limitação, incluindo o efeito
  sobre o poder estatístico da análise por estrato.
- [ ] Qualquer decisão de usar `top_k` fixo ou um único LLM está declarada
  como limitação de escopo, não como escolha ótima incontestável.
- [ ] A licença CC BY-NC 4.0 do HateBRXplain e suas implicações estão
  mencionadas.

### 12. Ética

- [ ] O texto reconhece que o corpus contém linguagem ofensiva real, de
  pessoas reais (políticos brasileiros).
- [ ] Exemplos individuais do dataset citados no TCC seguem a confirmação
  de licença já feita (12_decisoes.md: permitido, com atribuição, uso não
  comercial) — manter o número de exemplos pequeno e sempre citar a
  fonte.
- [ ] O processo de anotação de sarcasmo descreve quem anotou (autor +
  segundo anotador já confirmado) e como desacordos foram resolvidos.
- [ ] Riscos de uso indevido do classificador são discutidos, não apenas
  citados como problema de terceiros.

### 13. Apresentação/defesa

- [ ] Apresentação oral com duração entre 30 e 40 minutos (individual),
  conforme `0-regulamento.pdf`.
- [ ] Todo número apresentado nos slides tem lastro em
  `outputs/tables/*.json` ou `outputs/experiments/*.json`.
- [ ] Existe um exemplo concreto e reproduzível de classificação (com e
  sem RAG) para mostrar ao vivo ou em vídeo durante a defesa.
- [ ] O orientador (Raimundo Cláudio da Silva Vasconcelos) revisou e
  concordou com a versão final antes do envio à banca.

## Como usar este documento

Não é uma lista para marcar tudo de uma vez. É um alvo. Reveja:
- ao final da implementação de cada novo componente (RAG, LLM, anotação),
  para checar se algo aqui já pode ser marcado;
- antes de começar a escrever cada capítulo do TCC, para saber o que esse
  capítulo precisa comprovar;
- na revisão final, como última passada antes de enviar à banca.
