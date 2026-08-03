# Plano de preparação dos dados — fase 1 do TCC

Este plano transforma a etapa “Preparação e Ingestão de Dados” do PCC em entregas
reprodutíveis. A regra central é simples: nenhum comentário de validação ou teste
pode entrar no índice de recuperação, direta ou indiretamente.

## Etapa 0 — aquisição e proveniência

Registrar em um manifesto local ou relatório agregado:

- página e versão oficial do HateBRXplain;
- data de obtenção e nome original do arquivo;
- licença e restrições de redistribuição;
- hash SHA-256 do arquivo bruto.

O arquivo deve ficar em `data/raw/`, imutável e ignorado pelo Git. O repositório
público pode conter o procedimento de obtenção e o hash, mas não o corpus, salvo
se a licença autorizar explicitamente a redistribuição.

**Concluída quando:** outra pessoa consegue identificar exatamente a fonte e a
versão utilizadas sem receber uma cópia do corpus pelo repositório.

## Etapa 1 — contrato e auditoria do dataset

Executar a inspeção inicial sem alterar textos. Confirmar no arquivo real:

- nomes, tipos e significado das colunas;
- exatamente 7.000 registros e distribuição esperada de 3.500 por classe, ou
  documentar qualquer divergência;
- representação dos rótulos de ofensividade e dos *rationales*;
- valores ausentes, IDs existentes e linhas duplicadas;
- encoding e problemas de leitura.

Não remover caixa, acentos, emojis, pontuação, URLs ou repetições nesta fase.
Esses elementos podem carregar sinais linguísticos relevantes.

**Artefato:** `outputs/tables/dataset_audit.json`, contendo somente estatísticas
agregadas e seguro para revisão antes de eventual publicação.

**Concluída quando:** o esquema real foi documentado e toda divergência em relação
ao PCC tem uma decisão registrada em `docs/decisions.md`.

## Etapa 2 — IDs, duplicatas e dados derivados

Gerar um `comment_id` estável e não textual. Para detectar vazamento, comparar ao
menos:

1. duplicatas do texto original;
2. duplicatas após normalização usada apenas para comparação (Unicode, espaços e
   caixa), preservando o texto original no dado derivado;
3. rótulos conflitantes entre textos duplicados.

Duplicatas não devem ser simplesmente apagadas. Primeiro, formar grupos; depois,
decidir e registrar se cada grupo será mantido, consolidado ou excluído. Todos os
integrantes de um grupo devem permanecer na mesma partição.

**Artefatos:** dados intermediários em `data/interim/` e relatório agregado de
duplicatas em `outputs/tables/`.

**Concluída quando:** cada registro possui ID estável, não há decisão silenciosa de
limpeza e os grupos potencialmente vazantes estão identificados.

## Etapa 3 — partições congeladas

Criar treino, validação e teste com semente fixa, estratificação pelo rótulo de
ofensividade e agrupamento por duplicatas. A proporção deve ser aprovada após a
auditoria; uma opção inicial é 70/15/15, mas ela não fica definitiva antes de
verificar a distribuição e as necessidades da anotação de sarcasmo.

Persistir manifestos contendo apenas `comment_id`, partição e, se necessário,
rótulo. Implementar verificações automáticas para:

- ausência de IDs e grupos sobrepostos;
- preservação aproximada da distribuição das classes;
- reprodutibilidade com a mesma semente;
- uso exclusivo do treino no futuro índice RAG.

**Artefatos:** manifestos em `data/processed/` e relatório agregado das partições.

**Concluída quando:** os testes de integridade passam e as partições são congeladas
antes de ajustar prompts, `top-k`, embeddings ou modelos.

## Etapa 4 — piloto de sarcasmo

Somente depois de congelar as partições, selecionar do teste o piloto de 100 itens
previsto no protocolo, balanceado por ofensividade. Aplicar
`docs/annotation-guidelines.md`, revisar casos incertos e versionar o guia antes da
anotação principal.

As anotações contêm texto sensível e não devem ser publicadas automaticamente.
Publicar apenas estatísticas agregadas até confirmar licença, consentimento e
orientação institucional.

**Concluída quando:** o guia foi calibrado, os desacordos foram documentados e está
definido como os rótulos `incerto` entram ou não na análise.

## O que vem depois

Com as etapas anteriores aprovadas, a sequência técnica é:

1. baseline determinístico de classificação;
2. baseline do mesmo LLM sem recuperação;
3. escolha e avaliação do modelo de embeddings no conjunto de validação;
4. índice construído exclusivamente com treino;
5. RAG com `top-k` definido na validação;
6. execução única da avaliação final no teste e análise separada do estrato
   sarcástico.

Pinecone, LangChain e FastAPI são meios de implementação, não pré-requisitos para
validar a hipótese. Uma prova local simples deve anteceder a infraestrutura de
serviço.
