# Registro de decisões

## Histórico anterior ao PCC final (contexto, não lista de tarefas)

As pendências abaixo foram levantadas durante a revisão do PCC, antes da
versão final. A versão de 22 de julho de 2026 confirmou que os pontos
metodológicos centrais foram incorporados; a lista fica aqui só como
contexto histórico de como o tema evoluiu até ser aprovado.

- Corrigir a tabela que apresentava `Ofensivo / Sarcasmo` como uma única classe.
- Explicar que os *rationales* justificam ofensividade.
- Alinhar título, resumo, objetivo e saída: o texto alternava entre detectar
  sarcasmo, detectar ódio e classificar ofensividade em casos sarcásticos.
- Incluir protocolo do rótulo de sarcasmo e tratamento de incerteza.
- Definir como os casos sarcásticos seriam selecionados para avaliação.
- Impedir indexação de itens do teste na base RAG.
- Especificar baselines, ablações, métricas por estrato e análise estatística.
- Tratar LangChain/Pinecone como decisões de implementação, sujeitas a revisão.
- Padronizar `sarcasmo`/`ironia` conforme a definição operacional.
- Revisar objetivos específicos e a caracterização da revisão da literatura.
- Conferir números, resultados e referências nas fontes primárias.

Para o mapeamento atualizado entre o que o PCC promete e o que o projeto
efetivamente implementa, ver [13_guia_escrita_tcc.md](13_guia_escrita_tcc.md).

## 2026-07-20 — Nome do projeto

- Decisão: usar `rag-hatespeech-ptbr`.
- Motivo: nome escolhido pelo autor para o repositório em `github.com/efdourado`.

## 2026-07-20 — Separação entre ofensividade e sarcasmo

- Status: consolidada na versão final do PCC de 22 de julho de 2026.
- Decisão: tratar ofensividade como alvo e sarcasmo como estrato de avaliação.
- Motivo: HateBRXplain fornece rótulos e *rationales* de ofensividade, não rótulos
  verdadeiros de sarcasmo.

## 2026-07-20 — Dependências de RAG

- Decisão: adiar LangChain, Pinecone e clientes de LLM.
- Motivo: primeiro será auditado o dataset e congelado o protocolo.

## 2026-07-22 — Ordem de implementação no TCC

- Decisão: concluir aquisição, contrato, auditoria, identificação e divisão dos
  dados antes de implementar embeddings, Pinecone, LangChain, FastAPI ou LLM.
- Motivo: a divisão define quais exemplos podem entrar no índice RAG e evita
  vazamento experimental; escolhas de infraestrutura não devem anteceder a
  validação do corpus.

## 2026-07-22 — Imutabilidade dos dados brutos

- Decisão: preservar o arquivo oficial em `data/raw/` sem alterações e produzir
  quaisquer correções somente em `data/interim/` ou `data/processed/`.
- Motivo: garantir rastreabilidade, reprodutibilidade e comparação com a fonte.

## 2026-08-04 — Fonte e versão do HateBRXplain

- Decisão: usar `dataset/HateBRXplain.csv` do repositório mantido por Francielle
  Vargas, fixado na revisão `0ac05461f17c4d7b7655fbee0391ec325e61a83b`.
- Integridade: SHA-256
  `dd2b669ab8c055805b4813b2628961d951273577fc637039f5eec1238cc9f8bb`.
- Licença adotada: CC BY-NC 4.0, conforme o repositório de origem.
- Motivo: a versão consolidada possui IDs, nomes de coluna estáveis, links das
  publicações para as duas classes e uma declaração explícita de licença.

## 2026-08-04 — Resultado da auditoria inicial

- Resultado: aprovado em todos os controles automatizados.
- Evidências agregadas: 7.000 registros e IDs únicos; 3.500 itens por classe;
  nenhum comentário duplicado exato ou após normalização comparativa; dois
  *rationales* presentes nos 3.500 itens ofensivos e ausentes nos não ofensivos.
- Publicações de origem: 85 valores distintos de `link_post`, dos quais 53
  aparecem nas duas classes.
- Artefato: `outputs/tables/dataset_audit.json`.

## 2026-08-04 — Divisão agrupada e congelada

- Decisão: manter todos os comentários do mesmo `link_post` canônico na mesma
  partição, mesmo que isso impeça uma proporção exatamente igual a 80/10/10.
- Método: `StratifiedGroupKFold` com 10 folds, embaralhamento, semente 42 e
  scikit-learn 1.9.0; fold 0 reservado para teste, fold 7 para validação e os
  demais para treino.
- Resultado: treino com 5.608 itens (2.804 por classe), validação com 706 (353 por
  classe) e teste com 686 (343 por classe).
- Isolamento: 67 publicações no treino, 6 na validação e 12 no teste, sem
  sobreposição de IDs, publicações ou comentários normalizados.
- Manifesto: `data/processed/split_manifest.csv`, contendo apenas `id`,
  `fold` e `split`; SHA-256
  `6a7212fbb42c26fc8e9c8f1ef3fe391fb9f651d5c14820fc3e2b521002208033`.
- Política RAG: somente IDs marcados como `train` podem ser indexados ou
  recuperados como contexto.
- Motivo: reduzir vazamento temático entre comentários associados à mesma
  publicação, ainda que a divisão deixe de ser diretamente comparável à divisão
  por item usada no artigo original.

## 2026-08-04 — Limitação de concentração por publicação

- Constatação: o maior `link_post` representa 35,6% da validação e 45,9% do teste;
  os três maiores representam, respectivamente, 87,5% e 73,9%.
- Consequência: comentários de uma mesma publicação não serão tratados como
  observações estatisticamente independentes.
- Decisão analítica: intervalos e comparações pareadas devem reamostrar por
  `link_post`; também será reportada uma análise de sensibilidade sem a maior
  publicação do teste.
- Regra: os folds não poderão ser trocados após observar resultados de modelos.

## 2026-08-04 — Piloto de anotação fora do teste

- Decisão: calibrar o guia de sarcasmo com 100 itens da validação, balanceados por
  ofensividade, e congelar o guia antes da anotação do teste.
- Motivo: impedir que decisões de anotação sejam ajustadas após examinar o conjunto
  destinado à avaliação final.

## 2026-08-15 — Baseline determinístico

- Decisão: implementar o primeiro baseline de "O que vem depois"
  (10_plano_dados.md) como TF-IDF (uni+bigramas, `min_df=2`,
  `sublinear_tf=True`, acentos preservados) + regressão logística
  (`class_weight="balanced"`, semente 42), ambos do scikit-learn já
  presente nas dependências do projeto.
- Motivo: fornece um piso de comparação determinístico, sem custo de API e
  sem depender de decisões ainda em aberto na época (provedor de
  LLM/embeddings), respondendo diretamente ao objetivo específico "analisar
  as limitações de classificadores textuais diante de comentários curtos,
  sarcásticos e ambíguos" do PCC (1.5.2).
- Resultado na validação (706 itens, 5.608 de treino): F1 da classe
  ofensiva = 0,7646; precisão = 0,7701; revocação = 0,7592; macro-F1 =
  0,7663. Matriz de confusão `[[273, 80], [85, 268]]` (linhas/colunas na
  ordem `[não ofensivo, ofensivo]`).
- Artefatos: `outputs/tables/baseline_validation.json` (versionado);
  registro bruto individual em `outputs/experiments/` (não versionado, por
  `.gitignore`).
- Não avaliado no teste. Ver decisão seguinte.

## 2026-08-15 — Política de execução única no conjunto de teste

- Decisão: `scripts/run_baseline.py` (e qualquer script de avaliação
  futuro que seguir o mesmo padrão) só avalia no split de teste se o
  usuário passar `--confirm-final-test-run` explicitamente; toda execução
  no teste é acrescentada a `outputs/experiments/test_evaluations_ledger.jsonl`.
- Motivo: o protocolo de pesquisa prevê "execução única da avaliação final
  no teste" (9_protocolo_pesquisa.md); um mecanismo apenas documental
  é fácil de esquecer ao rodar um script rapidamente. O bloqueio técnico
  torna o descumprimento visível e deliberado, não acidental.
- Consequência prática: o baseline determinístico não foi avaliado no
  teste nesta sessão, de propósito — a avaliação no teste deve esperar
  até que o baseline de LLM e o RAG estejam prontos, para que todos os
  modelos sejam comparados no teste em uma única rodada final, conforme o
  protocolo.

## 2026-08-15 — Provedor de LLM: Gemini

- Decisão: usar a API do Gemini (Google) como provedor de LLM para os
  experimentos de classificação com e sem RAG, e considerar o modelo de
  embeddings da própria Gemini API para a etapa de recuperação.
- Motivo: acesso via assinatura paga do autor (Google AI Pro, plano
  estudante).
- Ressalva verificada por pesquisa: a assinatura de consumo (app/AI
  Studio) é um produto separado da cobrança por chamada da API
  programática — a API do Gemini tem camada gratuita própria (sem
  cartão), com limites de taxa por modelo. Plano: iniciar os experimentos
  na camada gratuita da API; só habilitar faturamento (Cloud Billing) se
  os limites de taxa da camada gratuita não comportarem a escala do
  projeto (706 itens de validação, 686 de teste, mais os exemplos
  recuperados por chamada). Confirmar limites atuais do modelo escolhido
  em ai.google.dev antes de rodar em lote.
- Pendente: escolher o modelo Gemini específico (geração e, se aplicável,
  embeddings) no momento da implementação, e reconfirmar as cotas
  gratuitas vigentes nesse momento.
- Fontes: [Billing — Gemini API](https://ai.google.dev/gemini-api/docs/billing),
  [Google AI Plans — Gemini API](https://ai.google.dev/gemini-api/docs/google-ai-plans).

## 2026-08-15 — Banco vetorial: tentar Pinecone primeiro

- Decisão: implementar a indexação com Pinecone como primeira tentativa;
  se surgir algum obstáculo real (custo inesperado, limite de conta,
  complexidade desproporcional), substituir por uma implementação local
  (`numpy`/`scikit-learn`, similaridade de cosseno em memória) e registrar
  a justificativa de que os resultados seriam equivalentes.
- Motivo: o plano gratuito do Pinecone comporta o projeto sem aperto — 2
  GB de armazenamento, até 300 mil vetores, 2 milhões de unidades de
  escrita e 1 milhão de unidades de leitura por mês, sem cartão de
  crédito. O treino tem 5.608 itens, muito abaixo de qualquer um desses
  limites. Além disso, a ferramenta tem valor de aprendizado para além
  deste TCC.
- Base da equivalência, caso seja necessário migrar para local: com
  apenas 5.608 vetores candidatos, busca exaustiva local e busca do
  Pinecone devem retornar os mesmos vizinhos mais próximos — a escolha
  entre as duas é de engenharia, não de resultado científico.
- Fontes: [Pinecone Database limits](https://docs.pinecone.io/reference/api/database-limits),
  [Free plan gets 3x more capacity with serverless upgrade](https://www.pinecone.io/blog/serverless-free/).

## 2026-08-15 — Orquestração: tentar LangChain primeiro

- Decisão: mesma lógica do banco vetorial — tentar LangChain para
  orquestrar recuperação + *prompt* + geração; se a abstração atrapalhar
  mais do que ajudar (dificultar explicar o fluxo exato na defesa, ou
  exigir contornos complicados), substituir por chamadas diretas à API
  com código próprio, justificando a troca pela equivalência de
  resultado.
- Motivo: diferente do Pinecone, custo não é o fator limitante aqui —
  LangChain é gratuito e de código aberto. O critério de troca é
  complexidade/organização, não dinheiro. Testar a ferramenta mais usada
  do mercado tem valor além deste TCC.

## 2026-08-15 — Incluir FastAPI no escopo

- Decisão: manter o serviço FastAPI descrito no PCC no escopo do TCC,
  implementado depois que o pipeline de experimentos (RAG completo)
  estiver funcionando.
- Motivo: há tempo disponível no cronograma restante para essa camada, e
  ela é útil como demonstração na defesa.

## 2026-08-15 — Esclarecimento do rótulo de sarcasmo `incerto`

- Esclarecimento: o rótulo `2` (incerto) em 11_diretrizes_anotacao.md não
  é uma terceira categoria semântica de sarcasmo — é o registro de que
  quem está anotando não tem, a partir do texto disponível, contexto
  suficiente para decidir com confiança entre sarcástico (1) e não
  sarcástico (0). É sobre a limitação da informação disponível, não sobre
  um tipo diferente de sarcasmo.
- Ponto de atenção: a anotação humana precisa continuar independente da
  previsão do classificador (já registrado em "Não fazer" de
  11_diretrizes_anotacao.md). Raciocinar "se o modelo com contexto não vai
  identificar como sarcasmo, eu também não identificaria, pelo mesmo
  motivo de falta de contexto" é uma intuição plausível sobre por que dois
  julgamentos poderiam coincidir, mas o rótulo humano deve continuar
  sendo produzido a partir da leitura direta do comentário (e do contexto
  realmente disponível a quem anota), nunca a partir de uma expectativa
  sobre a resposta do modelo.
- Pendente: a execução da anotação em si ainda não começou.

## 2026-08-15 — Segundo anotador humano disponível

- Decisão: um amigo do autor vai colaborar como segundo anotador para uma
  amostra da anotação de sarcasmo, permitindo calcular Cohen's kappa.
- Pendente: definir o tamanho da amostra dupla-anotada (o protocolo prevê
  10%–20%) quando a anotação principal começar.

## 2026-08-15 — Proposta de magnitude para a hipótese quantitativa

- Proposta: usando o tamanho do conjunto de teste já congelado (686
  itens, 343 por classe) e um cálculo de poder padrão para diferença de
  duas proporções (α = 0,05 bicaudal, poder = 80%, ambas as proporções em
  torno de 0,76 — âncora no F1 do baseline determinístico), a diferença
  mínima detectável é de aproximadamente 6 a 7 pontos percentuais,
  tratando o desenho como amostras independentes. Como o desenho real é
  pareado (o mesmo item de teste avaliado por dois modelos), um teste
  pareado (ex.: McNemar) tende a detectar diferenças menores que essa com
  o mesmo tamanho de amostra — o valor acima é um teto conservador, não
  uma estimativa precisa.
- Status: proposta de referência, não uma hipótese fechada. Precisa de
  confirmação do orientador antes de virar critério de sucesso, e deve
  ser recalculada com dados reais assim que o baseline de LLM sem RAG
  estiver rodando.
- Motivo de propor agora: definir o critério antes de olhar o resultado
  do teste evita que o critério de sucesso seja ajustado depois de ver o
  número, o que enfraqueceria qualquer alegação de melhoria.

## 2026-08-15 — Direção aceita para a análise de explicabilidade

- Decisão: a contribuição das justificativas recuperadas (objetivo
  específico 5 do PCC) será avaliada, pelo menos em parte, por um
  formulário/interface simples que coleta opiniões de um grupo de pessoas
  (colegas/amigos do autor) sobre a qualidade das justificativas geradas,
  com banco de dados (Postgres) e hospedagem gratuita (ex.: Vercel).
- Status: direção de desenho aceita; implementação adiada até existirem
  justificativas reais geradas pelo RAG para avaliar.
- Pendente: desenhar as perguntas do formulário, decidir tamanho da
  amostra de justificativas avaliadas e quantas pessoas participam,
  quando o RAG estiver produzindo saídas.

## 2026-08-15 — Fixação de versão do pandas adiada para o fim

- Decisão: manter `pandas>=2.2,<3` por enquanto; fixar a versão exata
  usada (hoje, 2.3.3) só na rodada final de experimentos que for para o
  TCC, já que o projeto não deve ser retomado depois da entrega.
- Motivo: não há necessidade prática agora; fixar cedo demais só custa
  ter que atualizar a restrição toda vez que o ambiente for recriado
  durante o desenvolvimento.

## 2026-08-15 — Versionar o manifesto de split

- Decisão: `data/processed/split_manifest.csv` passa a ser versionado no
  Git. Critério geral adotado: versionar todo artefato que não seja
  grande e que tenha valor para quem for revisar ou reproduzir o projeto
  sem rodar o pipeline inteiro.
- Motivo: o arquivo tem ~92 KB, contém apenas `id`, `fold` e `split`
  (nenhum texto de comentário), e permite conferir a partição sem rodar
  `scripts/create_splits.py`.
- Ação: `.gitignore` ajustado para manter a exclusão de
  `data/processed/*` em geral, com uma exceção explícita para este
  arquivo.

## 2026-08-15 — Confirmação sobre citar exemplos individuais do dataset

- Verificação feita: o README oficial do HateBR
  (`franciellevargas/HateBR`, que também hospeda o HateBRXplain) confirma
  a licença CC BY-NC 4.0 e declara explicitamente: "This dataset contains
  hateful and offensive content and is intended for research purposes
  only. Commercial use is not permitted." Não há, no README, nenhuma
  cláusula proibindo especificamente a citação de exemplos individuais em
  trabalhos acadêmicos.
- Conclusão informada (não é parecer jurídico): um TCC é uso não
  comercial por definição, e a CC BY-NC 4.0 permite explicitamente
  compartilhar e adaptar o material para fins não comerciais, com
  atribuição. Citar um pequeno número de exemplos ilustrativos, sempre
  com atribuição à fonte (VARGAS et al., 2022; SALLES; VARGAS;
  BENEVENUTO, 2025), segue a prática já estabelecida na área — o próprio
  HateXplain (MATHEW et al., 2021), citado no PCC, publica exemplos
  individuais do seu corpus.
- Decisão: citar exemplos individuais no TCC é permitido dentro dos
  termos da licença, com atribuição. Manter o número de exemplos citados
  pequeno e sempre com a citação da fonte, não apenas "extraído do
  dataset".
- Fontes: [franciellevargas/HateBR](https://github.com/franciellevargas/HateBR),
  [HateBR/README.md](https://github.com/franciellevargas/HateBR/blob/main/README.md).

## 2026-08-15 — Ferramenta de preparação do piloto de sarcasmo

- Decisão: implementar `scripts/prepare_sarcasm_pilot.py` para sortear os
  100 itens do piloto (50 por classe de ofensividade, só do split de
  validação, semente 42) e gerar duas planilhas `.xlsx` prontas para
  anotação manual: uma com os 100 itens (para o autor) e uma com um
  subconjunto de ~15 itens (para o segundo anotador), além de um relatório
  agregado sem texto em `outputs/tables/sarcasm_pilot_audit.json`.
- Motivo: o autor relatou não saber como começar a anotação manual. Uma
  ferramenta que sorteia a amostra corretamente (sem vazar itens de
  teste, sem expor `offensive_label`, de forma determinística e
  reprodutível) e entrega o resultado em formato de planilha comum reduz
  o atrito de começar, sem substituir o julgamento humano em nenhuma
  decisão de rótulo.
- Execução real: rodado nesta sessão contra o dataset local. Resultado:
  100 itens no piloto (50/50), 15 no subconjunto do segundo anotador,
  ambos amostrados exclusivamente do split `validation`. Reexecutado uma
  segunda vez para confirmar reprodutibilidade — relatório agregado saiu
  byte-idêntico.
- Escopo do que a ferramenta faz e não faz: só sorteia a amostra e
  formata as planilhas. Não anota nada, não sugere rótulos (nenhuma
  pré-anotação por IA está implementada), e não calcula concordância
  entre anotadores — isso fica para uma etapa futura, depois que as
  planilhas estiverem preenchidas.
- Artefatos locais (não versionados, contêm texto de comentários):
  `data/annotations/sarcasm_pilot.xlsx`,
  `data/annotations/sarcasm_pilot_second_annotator.xlsx`.

## 2026-08-15 — Resultado real do piloto de sarcasmo e risco de poder estatístico

- Resultado observado (planilha principal, 100 itens da validação): 84
  `0` (não sarcástico), 15 `2` (incerto), 1 `1` (sarcástico). Segundo
  anotador (15 itens, subconjunto dos mesmos 100): 13 `0`, 2 `1`, 0 `2`.
  Concordância exata nos 15 itens em comum: 80% (12/15); Cohen's kappa:
  0,196 — baixo apesar da concordância bruta alta, efeito esperado quando
  a distribuição é muito desbalanceada (poucos casos fora de `0`).
  Calculado por `scripts/check_annotation_agreement.py`, registrado em
  `outputs/tables/sarcasm_annotation_agreement.json`.
- Constatação: taxa de sarcasmo claramente identificado é baixa
  (~1% na amostra). Coerente com a natureza do corpus — os comentários do
  HateBRXplain (auditados desde 2026-08-04) são majoritariamente ofensa
  direta ("lixo", "comunista safada", "vagabunda"), não ironia. Se essa
  taxa se mantiver no teste (686 itens), o estrato sarcástico teria da
  ordem de poucas unidades a poucas dezenas de itens — pouco poder
  estatístico para uma comparação quantitativa de F1 por estrato.
- O que isso NÃO afeta: o experimento principal (RAG vs. Gemini sem RAG,
  nos 686 itens do teste, sobre `offensive_label`) independe da taxa de
  sarcasmo — o índice RAG é construído a partir de ofensividade no
  treino, nunca de rótulos de sarcasmo. Só a análise secundária
  ("melhora mais nos casos sarcásticos?") está em risco de ficar
  subdimensionada.
- Decisão: ampliar a amostra de calibração dentro da validação antes de
  decidir qualquer mudança de escopo — sorteio aleatório balanceado por
  ofensividade do restante do split (nunca curadoria manual de itens
  "que parecem sarcásticos", e nunca usando os *rationales* de
  ofensividade como pista, pelo mesmo motivo de sempre: eles justificam
  ofensividade, não sarcasmo). Implementado em
  `scripts/expand_sarcasm_sample.py`.
- Execução real: gerado um segundo lote com os 606 itens restantes da
  validação (303 por classe — todo o restante do split, já que o autor
  sinalizou disposição de anotar bastante mais que os 100 iniciais),
  excluindo os 100 já em circulação. Verificado que nenhum id do novo
  lote pertence a treino ou teste, e que a execução é reprodutível
  (relatório byte-idêntico ao reexecutar). Artefato local (não
  versionado): `data/annotations/sarcasm_pilot_batch2.xlsx`. Relatório
  agregado (sem texto): `outputs/tables/sarcasm_sample_expansion_audit.json`.
- Pendente, precisa de confirmação do orientador: se a taxa de sarcasmo
  continuar baixa mesmo com a amostra ampliada, o objetivo específico 5
  do PCC ("melhora... especialmente em comentários sarcásticos") pode
  precisar ser reportado como análise qualitativa/exploratória dos casos
  encontrados, em vez de comparação estatística de F1 por estrato. Essa
  mudança de escopo não deve ser decidida unilateralmente — precisa
  constar como decisão registrada, com a anuência do orientador, antes de
  entrar no texto final do TCC.

## 2026-08-15 — Anotação de calibração solo + checagem de confiabilidade com múltiplos verificadores

- Decisão: em vez de distribuir pedaços exclusivos dos 606 itens do
  `batch2` entre vários amigos, o autor anota tudo sozinho. Motivo:
  distribuir pedaços não sobrepostos entre pessoas não treinadas
  reproduz (e pode piorar) o mesmo problema já visto no primeiro
  segundo anotador — cada pessoa aplica o guia à sua maneira, sem
  nenhuma checagem cruzada possível na maior parte dos itens.
- Decisão complementar: em vez de mais volume, ampliar a *confiabilidade*
  do que já foi anotado. Implementado
  `scripts/prepare_reliability_check.py`: sorteia um subconjunto (30
  itens, semente 44) do que o autor já rotulou, exclui os 15 itens já
  checados pelo primeiro segundo anotador, e gera cópias idênticas e em
  branco — uma por verificador — para pessoas diferentes julgarem
  independentemente os mesmos itens. Não é anotação nova: mede o quanto
  o critério do autor se sustenta com outras pessoas olhando o mesmo
  comentário.
- Implementado também `scripts/check_multi_rater_agreement.py` /
  `compute_multi_rater_agreement()`: kappa par a par (autor vs. cada
  verificador, reaproveitando `compute_agreement`) e, com 2+
  verificadores respondidos, Fleiss' kappa conjunto (implementado à mão,
  sem dependência nova, validado por dois casos calculados manualmente:
  concordância perfeita → 1,0; um caso com discordância → 0,3078, ambos
  cobertos por teste).
- Execução real: gerados `sarcasm_reliability_check_1.xlsx`, `_2.xlsx`,
  `_3.xlsx` (30 itens idênticos e em branco cada, mesmos ids nos três,
  confirmados reprodutíveis). Mensagem curta pronta para enviar aos
  verificadores em
  `data/annotations/mensagem_para_verificadores.txt` (versionada — sem
  texto de comentário, só instrução).
- Bug encontrado e corrigido durante a implementação: concatenar uma
  planilha já rotulada (inteiros) com uma ainda em branco (lida do Excel
  como `NaN`) faz o pandas promover a coluna inteira para `float64`; uma
  comparação de string ingênua (`"0.0" != "0"`) descartava rótulos
  válidos como se estivessem em branco. Corrigido comparando
  numericamente (`pd.to_numeric(...).isin([0, 1, 2])`); teste de
  regressão adicionado.
- Reservado para o futuro (já registrado antes, reafirmado aqui): quando
  chegar a hora de anotar o conjunto de teste (686 itens, os que geram
  os números do TCC), repetir o esquema de segundo anotador em 10-20%
  da anotação, para haver um kappa reportável nos dados que sustentam a
  conclusão final — o lote de calibração atual (`batch2`) não precisa
  disso porque não vira resultado citado no texto.

## 2026-10-02 — Preservar anotações e iniciar OpenRouter/RAG

- O autor concluiu os 606 itens do segundo lote e decidiu não revisar novamente
  essa planilha. Preservar as duas planilhas originais e registrar seus hashes,
  sem completar evidências/notas ou substituir julgamentos humanos com IA.
  Validação completa: 627 rótulos `0`, 18 `1`, 61 `2` (706 itens).
  Snapshot agregado em `outputs/tables/sarcasm_calibration_snapshot.json`.
- Incertos permanecem incluídos no F1 geral de ofensividade e são reportados
  separadamente nos estratos. A anotação do teste e a abrangência da análise
  secundária ainda precisam ser tratadas antes da avaliação final.
- Por solicitação do autor, adotar OpenRouter no lugar da API direta de Gemini.
  A chave existe, mas deve ser usada somente posteriormente. Nenhuma inferência
  real pelo OpenRouter nesta sessão; o catálogo público foi consultado sem chave.
- Priorizar embeddings locais para o início sem cobrança de API; alternativa
  OpenRouter também implementada. Modelo multilíngue inicial e revisão fixados
  no relatório do índice: 5.608 vetores de 384 dimensões, somente do treino.
  Há 34 textos acima do limite de 128 tokens do encoder; auditar a influência
  dessa truncagem na validação antes de congelar o experimento final.
- Implementar índice local de referência, mesmo prompt para baseline/RAG/ablação,
  parsing estrito, cache, retomada, registro das execuções e FastAPI local.
  Chamadas diretas permitem inspecionar o payload e o uso nesta etapa. Pinecone
  e LangChain continuam sem implementação; não registrar uma tentativa/falha
  ou equivalência de índices que não ocorreu.
- Reservar o teste; desenvolver/calibrar na validação. Próximo passo real:
  escolher um LLM fixo compatível com a conta gratuita e comparar cinco itens.
  Guia operacional: `docs/14_execucao_openrouter_rag.md`.
