# Estado do projeto

> Ponto de entrada para orientar a continuidade do TCC. Este arquivo é
> vivo: deve ser atualizado sempre que um componente mudar de estado.
> Ele reflete o estado do repositório local, que está à frente do que
> qualquer leitura isolada do PCC (`2-pcc-final.pdf`) ou do README sugeriria.
> Última atualização: 2026-08-15.

## 1. Objetivo atual do TCC

Avaliar se uma abordagem baseada em RAG (recuperação de exemplos e
*rationales* de ofensividade do HateBRXplain, injetados no prompt de uma LLM)
melhora a classificação de ofensividade em português brasileiro, em
comparação com o mesmo LLM sem recuperação — com atenção especial ao
subconjunto de comentários sarcásticos.

Fonte: PCC final, 22/07/2026 (`2-pcc-final.pdf`), seções 1.4 e 1.5.1.

## 2. Pergunta de pesquisa

> "O uso de exemplos contextuais e *rationales* recuperados do conjunto de
> treino melhora a classificação de ofensividade, especialmente em
> comentários sarcásticos, quando comparado ao mesmo LLM sem recuperação?"

Fonte: 9_protocolo_pesquisa.md. Esta pergunta é uma formalização
operacional do objetivo geral do PCC (1.5.1) — não é um novo tema, é o mesmo
tema tornado testável.

## 3. Hipótese

Não há ainda uma hipótese estatística fechada (ex.: "espera-se que o F1 da
classe ofensiva aumente em pelo menos X pontos"). Existe uma expectativa
direcional no PCC/protocolo, e uma proposta de magnitude de referência
calculada em 12_decisoes.md ("Proposta de magnitude para a hipótese
quantitativa", ~6–7 pontos percentuais como teto conservador), ainda
pendente de confirmação do orientador e de recálculo com dados reais do
baseline de LLM sem RAG (ver 5_decisoes_pendentes.md, D7).

## 4. Ponto conceitual central (fixado)

- `offensive_label` é o alvo principal de classificação.
- Sarcasmo não é um rótulo nativo do HateBRXplain nem dos seus
  *rationales*.
- Sarcasmo é uma variável de análise/estrato, produzida por anotação humana
  separada, sobre uma amostra do conjunto de validação (piloto) e do
  conjunto de teste (principal).
- Os *rationales* (`rationales_annotator1`, `rationales_annotator2`)
  justificam ofensividade, não sarcasmo.
- Nenhum item de validação ou teste pode entrar no índice de recuperação
  (RAG), direta ou indiretamente.

Este ponto está consolidado em quatro lugares independentes e mutuamente
consistentes: PCC final (1.1, 1.6, 5.2), 12_decisoes.md,
9_protocolo_pesquisa.md e 11_diretrizes_anotacao.md.

## 5. Metodologia planejada (resumo)

1. Auditar o dataset bruto sem alterar textos. **Concluído.**
2. Gerar IDs estáveis, checar duplicatas exatas/normalizadas e conflitos de
   rótulo. **Concluído.**
3. Dividir treino/validação/teste com `StratifiedGroupKFold`, agrupando por
   `link_post` canônico, semente fixa. **Concluído, congelado.**
4. Calibrar o guia de sarcasmo com piloto de 100 itens da validação.
   **Pendente — requer anotação humana; segundo anotador já confirmado.**
5. Anotar sarcasmo no teste (IA apenas como pré-anotadora, decisão final
   humana). **Pendente — bloqueado pela etapa 4.**
6. Baseline determinístico de classificação de ofensividade. **Concluído**
   (ver seção 7).
7. Baseline do mesmo LLM (Gemini) sem recuperação. **Não iniciado —
   provedor já decidido, falta implementar.**
8. Escolha e avaliação de embeddings na validação. **Não iniciado —
   Gemini escolhido como ponto de partida, falta implementar.**
9. Índice RAG construído exclusivamente com o treino. **Não iniciado —
   Pinecone será tentado primeiro (ver 7_arquitetura_pipeline.md).**
10. RAG com `top-k` definido na validação. **Não iniciado.**
11. Execução única da avaliação final no teste + análise por estrato
    sarcástico. **Não iniciado — depende de 5, 7–10 estarem prontos.**

## 6. O que já está concluído

| Componente | Estado | Evidência |
|---|---|---|
| Aquisição e proveniência do HateBRXplain | Concluído | 12_decisoes.md (2026-08-04), hash SHA-256 verificado, `data/raw/HateBRXplain.csv` presente localmente |
| Auditoria de esquema/classes/duplicatas | Concluído | `scripts/inspect_dataset.py`, `outputs/tables/dataset_audit.json`, todos os `checks` em `true` |
| IDs estáveis e checagem de duplicatas | Concluído | mesmo relatório acima: `unique_ids: 7000`, zero duplicatas exatas ou normalizadas |
| Split reprodutível treino/validação/teste | Concluído e congelado | `src/rag_hatespeech_ptbr/splits.py`, `scripts/create_splits.py`, `outputs/tables/split_audit.json`, todos os `checks` em `true` |
| Testes automatizados da auditoria e do split | Concluído | `tests/test_inspect_dataset.py`, `tests/test_create_splits.py` — 7/7 passam |
| Lint | Limpo | `ruff check .` → "All checks passed!" |
| Reprodutibilidade ponta a ponta | Verificada | `inspect_dataset.py` e `create_splits.py` reexecutados contra o CSV local; `dataset_audit.json`, `split_audit.json` e `split_manifest.csv` saíram byte-idênticos aos versionados |
| Baseline determinístico | Concluído | ver seção 7 |
| Decisões de arquitetura (provedor de LLM, banco vetorial, orquestração, FastAPI) | Tomadas | 12_decisoes.md, entradas de 2026-08-15 |

## 7. Baseline determinístico

- Implementação: TF-IDF + Regressão Logística, treinado exclusivamente no
  conjunto de treino e avaliado na validação. Ver 6_guia_codigo.md para
  detalhes e 1_relatorio_sessao.md para os números completos.
- Infraestrutura reutilizável de carregamento de dados (`data.py`) e de
  métricas (`metrics.py`), pensada para ser reaproveitada pelos baselines de
  LLM e RAG futuros, evitando duplicar lógica de avaliação a cada novo
  modelo.
- Registro de experimentos (JSON por execução em `outputs/experiments/`),
  com um mecanismo explícito que impede a avaliação silenciosa no conjunto
  de teste.

## 8. O que está parcialmente concluído

- **Protocolo de pesquisa** (9_protocolo_pesquisa.md): metodologia
  definida; "pontos abertos" já revisados para refletir o PCC final e as
  decisões de 2026-08-15.
- **Guia de anotação de sarcasmo**: protocolo e rótulos definidos
  (11_diretrizes_anotacao.md), mas nunca aplicado a nenhum item real.
  Zero anotações existem em `data/annotations/` (pasta contém apenas
  `.gitkeep`).
- **Arquitetura do sistema RAG**: descrita conceitualmente no PCC (Pinecone
  + LangChain + FastAPI); a decisão atual é tentar as três (ver
  7_arquitetura_pipeline.md), mas nenhum código de recuperação, indexação,
  prompt ou chamada de LLM existe ainda no repositório.

## 9. O que ainda falta (lista objetiva)

1. Executar o piloto de anotação de sarcasmo (100 itens da validação) —
   tarefa humana, com um segundo anotador já confirmado para uma amostra.
2. Congelar o guia de anotação com base no piloto.
3. Anotar sarcasmo no conjunto de teste.
4. Escolher a versão exata do modelo Gemini (geração e embeddings) e
   confirmar a cota gratuita vigente no momento da implementação.
5. Implementar baseline do Gemini sem recuperação (com modo *dry-run*
   primeiro).
6. Implementar indexação vetorial restrita ao treino (Pinecone, com
   *fallback* local se necessário).
7. Implementar RAG (recuperação + prompt + geração + justificativa), via
   LangChain como primeira tentativa.
8. Executar a avaliação final única no teste, incluindo o estrato
   sarcástico.
9. Implementar o serviço FastAPI e a interface de avaliação de
   explicabilidade (depois dos itens acima).
10. Escrever o corpo do TCC (o PCC cobre introdução, fundamentação e
    revisão da literatura; metodologia, resultados, discussão e conclusão
    ainda não foram redigidos como texto corrido — apenas como
    planejamento técnico em `docs/`).
11. Preparar apresentação de defesa (mínimo 30, máximo 40 minutos, conforme
    `0-regulamento.pdf`).

## 10. Bloqueadores

| Bloqueador | Tipo | Impacto |
|---|---|---|
| Piloto e anotação principal de sarcasmo exigem julgamento humano | Estrutural | Bloqueia toda a análise estratificada por sarcasmo (o segundo eixo central do TCC) |
| Confirmação da cota gratuita do modelo Gemini escolhido | Verificação rápida no momento da implementação | Define se dá para rodar os experimentos sem habilitar faturamento |
| Confirmação do orientador sobre a magnitude da hipótese quantitativa (D7) | Decisão do orientador | Afeta o critério de sucesso da comparação final |

O restante das decisões que bloqueavam o projeto em versões anteriores
deste documento (provedor de LLM, Pinecone vs. local, LangChain vs. código
próprio, escopo do FastAPI, segundo anotador, citação de exemplos do
dataset) já foi resolvido — ver 12_decisoes.md, entradas de 2026-08-15.

## 11. Estimativa por componente

Estimativas qualitativas de esforço restante, não de tempo de calendário.
Cada linha é independente; não somam 100% porque os componentes têm peso
desigual no TCC final.

| Componente | Progresso estimado |
|---|---|
| Aquisição, contrato, auditoria do dataset | 100% |
| IDs, deduplicação, split reprodutível | 100% |
| Guia de anotação de sarcasmo (protocolo) | 100% (protocolo pronto; execução em 0%) |
| Anotação de sarcasmo (piloto + teste) | 0% |
| Baseline determinístico | 100% |
| Decisões de arquitetura (provedor, banco vetorial, orquestração) | 100% (decididas; implementação em 0%) |
| Baseline de Gemini sem recuperação | 0% |
| Seleção/avaliação de embeddings | 0% |
| Indexação vetorial (RAG) | 0% (dados prontos — treino isolado — mas nenhum código de indexação existe) |
| RAG completo (retrieval + prompt + geração) | 0% |
| Avaliação final única no teste | 0% (depende de tudo acima) |
| Análise por estrato sarcástico | 0% (depende da anotação) |
| Interface de avaliação de explicabilidade | 0% (direção aceita, implementação adiada) |
| Serviço FastAPI | 0% (no escopo, implementação adiada) |
| Texto do TCC (metodologia/resultados/discussão/conclusão) | ~5% (apenas planejamento técnico em `docs/`, sem prosa formal ainda) |
