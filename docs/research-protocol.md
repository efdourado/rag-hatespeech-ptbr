# Protocolo de pesquisa — versão de trabalho v0.2

Desenho derivado da versão final do PCC de 22 de julho de 2026. Decisões
experimentais ainda abertas devem ser fechadas com o orientador antes da execução
final, sem alterar o tema definido no PCC.

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
3. Gerar IDs estáveis a partir do conteúdo e verificar duplicatas exatas e textos
   normalizados antes de separar os dados.
4. Agrupar duplicatas e separar treino, validação e teste de forma estratificada e
   reprodutível.
5. Persistir os IDs e verificar automaticamente a ausência de sobreposição entre
   partições.
6. Usar somente o treino como base de recuperação.
7. Selecionar um piloto balanceado de 100 itens do teste (50 por classe de
   ofensividade) para refinar o guia de sarcasmo.
8. Anotar sarcasmo no teste; usar IA somente como pré-anotadora.
9. Comparar o mesmo LLM com baseline sem recuperação, *few-shot* fixo (se viável),
   RAG e uma ablação sem *rationales* (se prazo e custo permitirem).
10. Avaliar no teste completo e nos estratos sarcástico/não sarcástico.

## Métricas previstas

- Principal: F1 da classe ofensiva.
- Complementares: precisão, revocação, macro-F1 e matriz de confusão.
- Anotação: distribuição, incerteza e Cohen's kappa em amostra anotada
  independentemente por duas pessoas, se houver segundo anotador.
- Intervalos de confiança e teste pareado serão definidos antes da execução final.

## Controles contra vazamento

- Fixar semente e persistir IDs das partições.
- Deduplicar ou agrupar duplicatas antes da divisão.
- Nunca recuperar exemplos da validação ou do teste.
- Não ajustar prompt ou hiperparâmetros pelo resultado final do teste.
- Registrar modelo, prompt, parâmetros, custo e data de cada execução.

## Pontos abertos

- Confirmar pergunta e hipóteses finais.
- Confirmar proporções da divisão após inspecionar os dados oficiais.
- Definir embeddings, LLM, armazenamento vetorial e orçamento.
- Definir tratamento dos casos `incerto` na análise principal.
- Confirmar disponibilidade de segundo anotador humano.
