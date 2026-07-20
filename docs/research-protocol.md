# Protocolo de pesquisa — rascunho v0.1

Desenho provisório, a revisar após a nova versão do PCC e a validação do orientador.

## Pergunta principal provisória

O uso de exemplos contextuais e *rationales* recuperados do conjunto de treino
melhora a classificação de ofensividade, especialmente em comentários sarcásticos,
quando comparado ao mesmo LLM sem recuperação?

## Variáveis e rótulos

- Alvo principal: `offensive_label`, fornecido pelo HateBRXplain.
- Variável de análise: `human_sarcasm_label`, criada por anotação humana.
- Evidência recuperada: exemplos de treino e seus *rationales* de ofensividade.
- O rótulo de sarcasmo não será inferido a partir dos *rationales*.

## Desenho experimental provisório

1. Auditar o dataset original sem transformar os textos.
2. Remover apenas problemas documentados e gerar IDs estáveis.
3. Separar treino, validação e teste de forma estratificada e reprodutível.
4. Verificar duplicatas e impedir vazamento entre as partições.
5. Usar somente o treino como base de recuperação.
6. Selecionar um piloto balanceado de 100 itens do teste (50 por classe de
   ofensividade) para refinar o guia de sarcasmo.
7. Anotar sarcasmo no teste; usar IA somente como pré-anotadora.
8. Comparar o mesmo LLM com baseline sem recuperação, *few-shot* fixo (se viável),
   RAG e uma ablação sem *rationales* (se prazo e custo permitirem).
9. Avaliar no teste completo e nos estratos sarcástico/não sarcástico.

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
