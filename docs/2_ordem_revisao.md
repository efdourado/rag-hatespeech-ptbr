# Ordem de revisão

Tempo estimado: 30–45 minutos para uma primeira passada.

1. **Leia 1_relatorio_sessao.md** (5 min) — resumo do que mudou, resultados,
   decisões tomadas. Se algo aqui já responder sua dúvida, os passos
   seguintes ficam mais rápidos.

2. **Leia 4_estado_projeto.md, seções 4 a 10** (5 min) — confirme que o
   ponto conceitual central (seção 4) continua correto e que o resumo do
   que está concluído/pendente bate com sua própria percepção do projeto.

3. **Confirme os testes e o lint** (2 min):
   ```bash
   source .venv/bin/activate
   python -m pytest
   ruff check .
   ```
   Esperado: `22 passed` e `All checks passed!`.

4. **Reproduza o baseline** (2 min):
   ```bash
   python -m scripts.run_baseline data/raw/HateBRXplain.csv
   ```
   Compare o `positive_f1` impresso com 0,7646
   (`outputs/tables/baseline_validation.json`).

5. **Confira o `git diff` dos arquivos modificados** (5 min), não só os
   criados:
   ```bash
   git diff README.md 9_protocolo_pesquisa.md 10_plano_dados.md
   ```

6. **Leia 5_decisoes_pendentes.md** (5 min) — o que ainda depende de você:
   confirmar a magnitude da hipótese quantitativa (D7) com o orientador, e
   os detalhes do formulário de explicabilidade (D8) quando o RAG existir.

7. **Percorra 3_checklist_revisao.md, seção CRÍTICO** (5 min) — marque os
   itens que já conferiu nos passos 2–6 acima; a maioria se sobrepõe.

8. **Se for implementar algo em cima disso**, leia 6_guia_codigo.md
   (seções 3 a 6) e 7_arquitetura_pipeline.md antes de escrever qualquer
   código novo — evita duplicar as utilidades de `data.py`/`metrics.py`
   já criadas.

9. **Quando for escrever o texto do TCC**, use 13_guia_escrita_tcc.md como
   referência de o que já tem evidência e como checklist de qualidade
   antes de considerar uma seção pronta.

10. **8_auditoria_vazamento.md** pode ficar para quando você for
    implementar o RAG — reveja as seções sobre recuperação e prompt antes
    de escrever esse código, não depois.
