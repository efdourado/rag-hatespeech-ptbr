# Checklist de revisão do autor

Interface principal entre você e o trabalho feito em cada sessão. Itens
divididos por nível de risco caso passem despercebidos. Mantenha
atualizado conforme os itens forem revisados ou deixarem de se aplicar.

## CRÍTICO — você precisa conferir pessoalmente

Decisões metodológicas, interpretação de resultados, afirmações
científicas, alterações em objetivos, tudo que toca anotação humana.

- [ ] **O ponto conceitual central continua correto para você.**
  Onde está: 4_estado_projeto.md, seção 4.
  Por que importa: é o eixo que sustenta todo o resto do projeto
  (ofensividade como alvo, sarcasmo como estrato, *rationales* só de
  ofensividade, nunca indexar validação/teste).
  Evidência a procurar: os quatro documentos citados na seção 4
  (12_decisoes.md, PCC final, 9_protocolo_pesquisa.md,
  11_diretrizes_anotacao.md) dizem exatamente a mesma coisa.
  Como saber se está correto: releia a seção 4 e confirme que é isso
  mesmo que você quer defender perante a banca.
  Recomendação atual: nenhuma mudança necessária — consistente em todos
  os documentos auditados.

- [ ] **Os hiperparâmetros do baseline determinístico são uma escolha
  razoável, não uma tentativa de "acertar" um número bonito.**
  Onde está: `src/rag_hatespeech_ptbr/baseline.py`, função `build_pipeline`.
  Por que importa: qualquer escolha de hiperparâmetro que pareça calibrada
  para maximizar F1 (em vez de ser uma escolha padrão razoável) mina a
  credibilidade do baseline como piso honesto de comparação.
  Evidência a procurar: `ngram_range=(1,2)`, `min_df=2`,
  `sublinear_tf=True` são escolhas comuns de literatura, não obtidas por
  busca de hiperparâmetros neste projeto (nenhuma busca foi feita).
  Como saber se está correto: se você (ou o orientador) preferir outros
  valores "de livro-texto", pode trocar livremente — não há nada aqui
  ajustado ao resultado.
  Recomendação atual: manter como está; documentar no TCC que não houve
  busca de hiperparâmetros nesta etapa (é um baseline, não o resultado
  principal).

- [ ] **A decisão de não avaliar no teste ainda é a que você quer.**
  Onde está: 12_decisoes.md, 2026-08-15, "Política de execução única no
  conjunto de teste".
  Por que importa: uma vez que o teste for avaliado, cada nova avaliação
  fica registrada permanentemente em
  `outputs/experiments/test_evaluations_ledger.jsonl` — vale confirmar
  antes, não depois.
  Evidência a procurar: o arquivo do *ledger* ainda não existe no
  repositório — prova de que o teste nunca foi tocado.
  Como saber se está correto: confirme que você concorda em esperar até o
  RAG e o Gemini sem RAG estarem prontos antes de rodar `--split test`.
  Recomendação atual: manter a espera — é o que o protocolo já prometia,
  só foi formalizado tecnicamente.

- [ ] **A magnitude proposta para a hipótese quantitativa (D7) precisa da
  confirmação do orientador.**
  Onde está: 12_decisoes.md, "Proposta de magnitude para a hipótese
  quantitativa"; pendência rastreada em 5_decisoes_pendentes.md.
  Por que importa: define o critério de sucesso da comparação final antes
  de o resultado existir — essencial para que a conclusão não pareça
  ajustada depois de ver o número.
  Evidência a procurar: o cálculo (~6–7 pontos percentuais, teto
  conservador de um cálculo de poder para duas proporções, n=686) está
  documentado com as premissas usadas.
  Como saber se está correto: discuta com o orientador se essa magnitude
  (ou outro critério) é adequada para a defesa.
  Recomendação atual: usar como ponto de partida, recalcular quando o
  baseline de Gemini sem RAG estiver rodando.

- [ ] **O guia de anotação de sarcasmo ainda reflete como você quer
  definir sarcasmo — e a distinção do rótulo `incerto`.**
  Onde está: 11_diretrizes_anotacao.md; esclarecimento em 12_decisoes.md
  ("Esclarecimento do rótulo de sarcasmo `incerto`").
  Por que importa: uma vez que o piloto começar, mudar a definição
  operacional invalida anotações já feitas.
  Evidência a procurar: a definição de sarcasmo (contraste entre sentido
  literal e intenção) espelha a definição do PCC (1.1); o rótulo `2`
  (incerto) representa falta de contexto suficiente para decidir, não uma
  terceira categoria de sarcasmo.
  Como saber se está correto: releia antes de anotar o piloto de 100
  itens, e mantenha a anotação independente de qualquer expectativa sobre
  a resposta do modelo.
  Recomendação atual: nenhuma mudança — lembrete de revisar antes de
  começar a anotar de verdade.

## IMPORTANTE — você deve revisar

Escolhas de implementação, estrutura textual, decisões de design que
afetam manutenibilidade, mas não mudam o resultado científico diretamente.

- [ ] **Estrutura do módulo de dados (`data.py`) e se a verificação de hash
  por padrão (`verify_hashes=True`) é o comportamento que você quer.**
  Onde está: `src/rag_hatespeech_ptbr/data.py`.
  Por que importa: torna impossível rodar um experimento sem querer sobre
  um dataset ou split não aprovado — mas também significa que qualquer
  atualização legítima do dataset vai quebrar o script até as constantes
  `EXPECTED_*` em `splits.py` serem atualizadas.
  Evidência a procurar: `tests/test_data.py` cobre os dois casos (hash
  batendo e não batendo).
  Como saber se está correto: rode `python -m scripts.run_baseline
  data/raw/HateBRXplain.csv` e confirme que funciona sem erro.
  Recomendação atual: manter — a fricção é proposital.

- [ ] **O padrão de registro de experimentos (JSON bruto em
  `outputs/experiments/`, canônico em `outputs/tables/`).**
  Onde está: `scripts/run_baseline.py`.
  Por que importa: é o padrão que todo experimento futuro (Gemini, RAG)
  deve seguir.
  Evidência a procurar: compare `outputs/tables/baseline_validation.json`
  com `outputs/tables/dataset_audit.json` — mesmo espírito de relatório
  agregado.
  Como saber se está correto: veja se os campos (hiperparâmetros, hashes,
  métricas) são suficientes para confiar no resultado meses depois.
  Recomendação atual: manter; ao adicionar Gemini/RAG, adicionar também
  campos de custo/latência de API ao registro.

- [ ] **7_arquitetura_pipeline.md — o plano de tentar Pinecone e LangChain
  primeiro, com *fallback* local se travar.**
  Onde está: 7_arquitetura_pipeline.md, itens 8, 9, 11.
  Por que importa: é a estratégia que evita gastar tempo em infraestrutura
  antes de confirmar que ela funciona sem dor de cabeça na conta gratuita.
  Evidência a procurar: os limites do plano gratuito do Pinecone (2 GB,
  300 mil vetores, 2M/1M unidades de escrita/leitura por mês) cobrem os
  5.608 itens de treino com folga — checável nas fontes citadas em
  12_decisoes.md.
  Como saber se está correto: ao implementar, confirmar que a conta
  gratuita realmente não pede cartão e não trava no volume do projeto.
  Recomendação atual: seguir o plano; documentar a substituição por local
  se for necessária.

- [ ] **12_decisoes.md — as novas entradas de 2026-08-15 refletem
  exatamente o que você decidiu na conversa.**
  Onde está: 12_decisoes.md, todas as entradas datadas de 2026-08-15.
  Por que importa: é o registro oficial que o TCC vai citar como
  justificativa de cada escolha de arquitetura.
  Evidência a procurar: compare cada entrada com o que você pediu.
  Como saber se está correto: se alguma decisão foi registrada de forma
  diferente do que você quis dizer, corrija o texto diretamente no
  arquivo.
  Recomendação atual: revisar uma vez, já que são a base de várias outras
  seções.

## MECÂNICO — pode ser verificado rapidamente

Formatação, testes passando, caminhos, nomes.

- [ ] **Testes passam.** Comando: `python -m pytest`. Esperado: `22
  passed`.
- [ ] **Lint limpo.** Comando: `ruff check .`. Esperado: `All checks
  passed!`.
- [ ] **Reprodutibilidade do dataset/split.** Comandos:
  `python scripts/inspect_dataset.py data/raw/HateBRXplain.csv` e
  `python -m scripts.create_splits data/raw/HateBRXplain.csv`. Esperado:
  saídas byte-idênticas às já versionadas.
- [ ] **Nenhum caminho absoluto específico de máquina no código.**
  Comando: `grep -rn "/home/" scripts/ src/ tests/`. Esperado: vazio.
- [ ] **Nenhuma credencial no repositório.** Verificação: `git diff
  --stat` e inspeção visual antes de comitar, especialmente quando o
  cliente Gemini/Pinecone entrar em cena (chaves de API).
- [ ] **`.gitignore` continua excluindo `outputs/experiments/*` e
  `data/raw|interim|annotations/*`, mas versiona
  `data/processed/split_manifest.csv` especificamente e não
  `outputs/tables/*`.**
- [ ] **Numeração e nomes dos arquivos em `docs/` batem com os links
  citados em todos os outros arquivos** (checagem útil depois de qualquer
  renomeação futura): `grep -rn "docs/[A-Z_]" docs/ README.md` não deve
  retornar nomes antigos em maiúsculas/hífen.
