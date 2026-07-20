# Guia de anotação de sarcasmo — rascunho v0.1

## Unidade de anotação

Um comentário completo, apresentado sem o rótulo de ofensividade e sem a sugestão
da IA durante a decisão humana inicial, sempre que o processo permitir.

## Rótulos

- `0` — não sarcástico: sentido pretendido compatível com o literal, ainda que o
  texto seja agressivo, ofensivo, exagerado ou humorístico.
- `1` — sarcástico: contraste identificável entre sentido literal e intenção,
  geralmente usado para crítica, ridicularização ou desprezo.
- `2` — incerto: decisão depende de contexto ausente, entonação ou conhecimento
  externo insuficiente.

Ironia e sarcasmo podem se sobrepor, mas o trabalho deve adotar uma definição
operacional única. Casos limítrofes serão discutidos durante o piloto.

## Campos

| Campo | Descrição |
|---|---|
| `comment_id` | identificador estável e não textual |
| `ai_sarcasm_label` | sugestão opcional da pré-anotação |
| `ai_confidence` | confiança declarada pela IA entre 0 e 1 |
| `human_sarcasm_label` | decisão humana final: 0, 1 ou 2 |
| `sarcasm_evidence` | menor trecho que sustenta a decisão |
| `annotation_notes` | justificativa curta para caso difícil |
| `review_status` | `pending`, `reviewed` ou `adjudicated` |

## Procedimento inicial

1. Fazer piloto de 100 comentários do futuro teste, balanceado por ofensividade.
2. Anotar sem consultar a sugestão da IA.
3. Comparar discordâncias, registrar exemplos e revisar este guia.
4. Congelar uma versão do guia antes da anotação principal.
5. Submeter 10% a 20% a segundo anotador humano independente, se disponível.

## Não fazer

- Não usar *rationale* de ofensividade como rótulo verdadeiro de sarcasmo.
- Não considerar toda ofensa implícita automaticamente sarcástica.
- Não substituir julgamento humano por resposta de LLM.
- Não alterar o rótulo por conhecer a previsão do sistema avaliado.
