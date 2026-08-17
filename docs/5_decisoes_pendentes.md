# Decisões ainda pendentes

A maior parte das decisões que estavam em aberto (provedor de LLM, Pinecone
vs. local, LangChain vs. código próprio, escopo do FastAPI, segundo
anotador, fixação de versão, versionamento do manifesto, citação de
exemplos do dataset) já foi resolvida — ver as entradas de 2026-08-15 em
[12_decisoes.md](12_decisoes.md). Este arquivo lista só o que continua
genuinamente em aberto.

## D7 — Confirmar a magnitude da hipótese quantitativa

Uma proposta de referência já foi calculada (diferença mínima detectável de
~6–7 pontos percentuais no F1, teto conservador de um cálculo de poder para
duas proporções independentes com n=686 — ver 12_decisoes.md,
"Proposta de magnitude para a hipótese quantitativa"). Falta:

- Confirmar com o orientador se essa magnitude é um critério de sucesso
  aceitável para a defesa, ou se outro critério é preferido.
- Recalcular com dados reais assim que o baseline de LLM sem RAG estiver
  rodando na validação — a proposta atual usa o F1 do baseline
  determinístico como âncora, não o do LLM, que será a comparação real.

Consequência de adiar além do razoável: se o critério só for fixado depois
de ver o resultado do teste, a análise perde rigor.

## D12 — Escopo da análise por estrato sarcástico, se a taxa continuar baixa

O piloto real (100 itens da validação) achou taxa de sarcasmo claramente
identificado de ~1%, com Cohen's kappa baixo (0,196) entre os dois
anotadores — ver 12_decisoes.md, "Resultado real do piloto de sarcasmo e
risco de poder estatístico". Um segundo lote de 606 itens (todo o
restante da validação) já foi gerado para reduzir a incerteza dessa
estimativa antes de decidir qualquer coisa.

Se a taxa continuar baixa mesmo com a amostra ampliada:

- Opção 1: manter o objetivo específico 5 do PCC como comparação
  estatística de F1 por estrato, aceitando que o intervalo de confiança
  vai ficar largo (documentado como limitação).
- Opção 2: reformular essa parte do TCC como análise qualitativa/estudo
  de caso dos exemplos sarcásticos encontrados, em vez de comparação
  quantitativa — mais honesto com o tamanho de amostra real, mas é uma
  mudança do que o texto promete.

Consequência de adiar: nenhuma agora (a amostra ampliada ainda precisa
ser anotada); mas precisa estar decidido antes de escrever o capítulo de
resultados do TCC.

## D8 — Desenho concreto do formulário de avaliação de explicabilidade

A direção já foi aceita (formulário/interface + Postgres + hospedagem
gratuita, coletando opiniões de um grupo de conhecidos sobre a qualidade das
justificativas geradas). Falta, quando o RAG já estiver produzindo saídas
reais:

- Definir as perguntas do formulário (ex.: escala de concordância com a
  justificativa? comparação lado a lado com o *rationale* humano recuperado?
  pergunta aberta?).
- Definir quantas justificativas cada pessoa avalia e quantas pessoas
  participam, para ter uma amostra que valha a pena analisar.
- Confirmar se as respostas precisam de algum tipo de consentimento
  registrado, mesmo informal, já que trata de conteúdo ofensivo real.

Não bloqueia nada agora — não há saída de RAG para avaliar ainda.

## Acompanhar (não são decisões, são resultados de tentativa)

- **Pinecone:** a decisão já é tentar primeiro. O que falta é apenas
  confirmar, ao implementar, que a conta gratuita funciona sem
  complicação para os ~5.608 vetores do treino. Se algo travar, a
  substituição por implementação local já está pré-aprovada em
  12_decisoes.md.
- **LangChain:** mesma lógica — decisão já tomada (tentar primeiro),
  só falta confirmar na prática que a abstração não atrapalha mais do
  que ajuda.
- **Modelo Gemini específico:** escolher a versão exata do modelo de
  geração (e de embeddings, se for usar o da Gemini) no momento da
  implementação, e reconfirmar a cota gratuita vigente nesse momento —
  cotas de API mudam com frequência.
