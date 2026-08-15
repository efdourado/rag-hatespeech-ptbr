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

O rótulo `2` não é uma terceira categoria semântica de sarcasmo — é o
registro de que quem está anotando não tem, a partir do texto disponível,
contexto suficiente para decidir com confiança entre `0` e `1`. A decisão
deve vir sempre da leitura direta do comentário (e do contexto realmente
disponível), nunca de uma expectativa sobre a resposta do classificador em
avaliação (ver 12_decisoes.md, "Esclarecimento do rótulo de sarcasmo
`incerto`").

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

1. Fazer piloto de 100 comentários da validação, balanceado por ofensividade.
2. Anotar sem consultar a sugestão da IA.
3. Comparar discordâncias, registrar exemplos e revisar este guia.
4. Congelar uma versão do guia antes da anotação principal.
5. Submeter 10% a 20% a segundo anotador humano independente (já
   confirmado — ver 12_decisoes.md, "Segundo anotador humano disponível").

O piloto de calibração não poderá usar itens do teste. Após o congelamento do guia,
os rótulos do teste serão produzidos sem novas mudanças nas regras motivadas pelos
resultados experimentais.

## Passo a passo do piloto (planilhas já geradas)

1. Gerar (ou regerar) as planilhas:

   ```bash
   source .venv/bin/activate
   python -m scripts.prepare_sarcasm_pilot data/raw/HateBRXplain.csv
   ```

   Isso cria dois arquivos em `data/annotations/` (locais, nunca
   versionados, porque contêm texto real dos comentários):

   - `sarcasm_pilot.xlsx` — os 100 itens (50 ofensivos, 50 não ofensivos,
     sorteados só do conjunto de validação, nunca do teste). É a planilha
     que você preenche.
   - `sarcasm_pilot_second_annotator.xlsx` — um subconjunto de ~15 itens
     (mesmos `id`s de 15 linhas do arquivo principal), para o segundo
     anotador preencher de forma independente.

   Também grava `outputs/tables/sarcasm_pilot_audit.json` (versionado —
   só ids e contagens, sem texto), para conferir depois que a amostra foi
   sorteada corretamente.

2. Abrir `data/annotations/sarcasm_pilot.xlsx` no Excel, Google Sheets ou
   LibreOffice Calc. Colunas:

   | Coluna | O que fazer |
   |---|---|
   | `id` | não mexer — é só o identificador do comentário |
   | `comment` | leia este texto: é o comentário a classificar |
   | `human_sarcasm_label` | preencha `0`, `1` ou `2` (menu suspenso já configurado na célula) — ver definições em "Rótulos" acima |
   | `sarcasm_evidence` | opcional: cole o menor trecho do comentário que sustenta sua decisão |
   | `annotation_notes` | opcional: uma frase curta se o caso for difícil |
   | `review_status` | deixe `pending`; mude para `reviewed` quando tiver certeza da linha |

3. Anote linha por linha, na ordem que preferir, sem pular a leitura de
   nenhum comentário. Não consulte nada além do próprio texto do
   comentário — nem o rótulo de ofensividade (que nem está na planilha),
   nem qualquer sugestão de IA (não existe pré-anotação de IA implementada
   ainda), nem o que você acha que um classificador diria.

4. Salve o arquivo (`Ctrl+S`/`Cmd+S`) periodicamente — pode anotar aos
   poucos, não precisa terminar tudo de uma vez.

5. Para o segundo anotador: envie `sarcasm_pilot_second_annotator.xlsx`
   para o seu amigo (e-mail, Drive, WhatsApp — como preferir; é um
   arquivo local, não tem nada automatizado de envio). Ele preenche a
   coluna `human_sarcasm_label` de forma independente, sem ver as suas
   respostas para os mesmos itens.

6. Quando as duas planilhas estiverem preenchidas, isso vira insumo para:
   revisar discordâncias e ajustar este guia (passo 3 do procedimento
   acima), calcular Cohen's kappa entre as duas anotações do subconjunto
   comum, e só depois congelar o guia para anotar o teste. O código para
   calcular a concordância e consolidar as respostas ainda não existe —
   é o próximo passo depois que houver planilhas preenchidas para
   processar.

Não é preciso "enviar" a planilha principal para lugar nenhum: ela fica
local, em `data/annotations/`, e o próprio repositório já a mantém fora do
Git (contém texto de comentários reais). O `sarcasm_pilot_audit.json`
(sem texto) é que fica versionado, como prova de que a amostra foi
sorteada de forma correta e reprodutível.

## Não fazer

- Não usar *rationale* de ofensividade como rótulo verdadeiro de sarcasmo.
- Não considerar toda ofensa implícita automaticamente sarcástica.
- Não substituir julgamento humano por resposta de LLM.
- Não alterar o rótulo por conhecer a previsão do sistema avaliado.
