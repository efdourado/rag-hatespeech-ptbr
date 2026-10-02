# Protocolo de pesquisa — versão de trabalho v0.3

> Atualização em 2026-10-02: OpenRouter substitui a integração direta com Gemini;
> o modelo de geração será escolhido antes da primeira inferência. Embeddings
> semânticos locais são o ponto de partida para evitar custo de API. Anotações
> de validação são preservadas como entregues; incertos são reportados à parte
> nos estratos e permanecem incluídos na métrica geral de ofensividade.
> Pergunta, splits, alvo e reserva do teste permanecem os mesmos.

Desenho derivado da versão final do PCC de 22 de julho de 2026. Decisões
experimentais ainda abertas devem ser fechadas com o orientador antes da execução
final, sem alterar o tema definido no PCC.

> Atualizado em 2026-08-15: a pergunta principal abaixo já está congelada
> desde a versão final do PCC (22/07/2026, seção 1.5.1) e não é mais um
> ponto em aberto. A lista de pontos abertos foi revisada para refletir
> apenas o que de fato ainda depende de decisão. Ver
> [`13_guia_escrita_tcc.md`](13_guia_escrita_tcc.md) para o
> mapeamento completo entre PCC e TCC, e
> [`5_decisoes_pendentes.md`](5_decisoes_pendentes.md) para as decisões que
> ainda precisam do autor/orientador.

## Pergunta principal

O uso de exemplos contextuais e *rationales* recuperados do conjunto de treino
melhora a classificação de ofensividade, especialmente em comentários sarcásticos,
quando comparado ao mesmo LLM sem recuperação?

## Variáveis e rótulos

- Alvo principal: `offensive_label`, fornecido pelo HateBRXplain.
- Variável de análise: `human_sarcasm_label`, criada por anotação humana.
- Evidência recuperada: exemplos de treino e seus *rationales* de ofensividade.
- O rótulo de sarcasmo não será inferido a partir dos *rationales*.

## Desenho experimental

1. Auditar o dataset original sem transformar os textos.
2. Preservar uma cópia bruta imutável; toda correção deve ocorrer em dados derivados
   e constar em relatório.
3. Validar os IDs fornecidos pela fonte e verificar duplicatas exatas e textos
   normalizados antes de separar os dados.
4. Separar treino, validação e teste com `StratifiedGroupKFold`, agrupando pela
   versão canônica de `link_post` e estratificando por ofensividade.
5. Persistir os IDs e verificar automaticamente a ausência de sobreposição entre
   partições.
6. Usar somente o treino como base de recuperação.
7. Selecionar um piloto balanceado de 100 itens da validação (50 por classe de
   ofensividade) para refinar e congelar o guia de sarcasmo antes de tocar o teste.
8. Anotar sarcasmo no teste; usar IA somente como pré-anotadora.
9. Comparar o mesmo LLM com baseline sem recuperação, *few-shot* fixo (se viável),
   RAG e uma ablação sem *rationales* (se prazo e custo permitirem).
10. Avaliar no teste completo e nos estratos sarcástico/não sarcástico.

## Métricas previstas

- Principal: F1 da classe ofensiva.
- Complementares: precisão, revocação, macro-F1 e matriz de confusão.
- Anotação: distribuição, incerteza e Cohen's kappa em amostra anotada
  independentemente por duas pessoas, se houver segundo anotador.
- Intervalos de confiança e comparações pareadas considerarão `link_post` como
  unidade de reamostragem, evitando tratar comentários da mesma publicação como
  observações totalmente independentes.

## Controles contra vazamento

- Fixar semente e persistir IDs das partições.
- Manter cada `link_post` canônico em uma única partição.
- Nunca recuperar exemplos da validação ou do teste.
- Não ajustar prompt ou hiperparâmetros pelo resultado final do teste.
- Registrar modelo, prompt, parâmetros, custo e data de cada execução.

## Pontos abertos

Atualizado em 2026-08-15 — a maior parte destes pontos foi resolvida; ver
12_decisoes.md para as decisões e 5_decisoes_pendentes.md para o que
continua em aberto.

- ~~Confirmar pergunta e hipóteses finais.~~ Resolvido: a pergunta principal
  está congelada desde o PCC final (22/07/2026). Uma proposta de magnitude
  para a hipótese quantitativa já foi calculada (12_decisoes.md), pendente
  de confirmação do orientador e de recálculo com dados reais.
- ~~Definir embeddings, LLM, armazenamento vetorial e orçamento.~~
  Resolvido: Gemini como provedor de LLM/embeddings; Pinecone e LangChain
  serão tentados primeiro, com *fallback* local se necessário
  (12_decisoes.md).
- ~~Definir se Pinecone/LangChain/FastAPI (nomeados no PCC) serão usados
  como descrito ou substituídos por uma implementação local mais
  simples.~~ Resolvido: tentar as ferramentas nomeadas no PCC primeiro;
  FastAPI incluído no escopo.
- Definir tratamento dos casos `incerto` na análise principal — ainda em
  aberto, depende dos dados do piloto de anotação.
- ~~Confirmar disponibilidade de segundo anotador humano.~~ Resolvido: um
  segundo anotador já está disponível.
