# Conjunto de teste holdout - acervo editorial

## Situação do acervo

A conferência das pastas `OneDrive_1_24-06-2026` e `OneDrive_2_24-06-2026`, realizada em 21 de julho de 2026, encontrou 101 diretórios e 100 documentos distintos. O documento `234851_TD 3183_Estoque de capital` aparece nas duas pastas com cinco arquivos idênticos, verificados por hash SHA-256. A cópia a preservar é `OneDrive_1_24-06-2026`; a cópia identificada para remoção é `OneDrive_2_24-06-2026`.

O relatório de padrões contabiliza 90 pastas processadas. Como a duplicata foi processável, isso corresponde a 89 documentos distintos usados no aprendizado. Há 11 documentos distintos que não entraram na extração automática, pois não possuem simultaneamente um arquivo `(original).docx` e um arquivo `(para diagramar).docx` ou `(sem marcas).docx`.

## Localização isolada

Foram criadas cópias isoladas, sem modificar os arquivos de origem, em `C:\Users\jeffe\Downloads\IPEA_editorial_holdout_teste`:

- `executaveis/`: os nove casos que possuem `original.docx` e podem ser revisados pelo pipeline;
- `incompletos/`: os dois casos sem `original.docx`, preservados apenas como referência do acervo.

## Candidatos de teste sem contaminação

Os nove itens abaixo possuem um `original.docx` e não participaram do aprendizado. Eles podem ser revisados pelo pipeline e comparados manualmente ao DOCX com marcas e/ou PDF final, mas ainda não podem alimentar a avaliação automática porque falta a versão final em DOCX.

| Documento | Pasta de origem | Limitação |
|---|---|---|
| BPS 30 - Educação | `OneDrive_1_24-06-2026/224186_Boletim_BPS 30 (Educação)` | Sem `sem marcas` ou `para diagramar` em DOCX |
| Livro Monitoramento e avaliação - capítulo 4 | `OneDrive_2_24-06-2026/235284_Livro_Monitoramento e avaliação (cap. 4)` | Sem `sem marcas` ou `para diagramar` em DOCX |
| BMT 80 - ES 2 | `OneDrive_2_24-06-2026/235784_Boletim_BMT 80 (ES 2)` | Sem `sem marcas` ou `para diagramar` em DOCX |
| PPP 71 - artigo 6 | `OneDrive_2_24-06-2026/236012_Revista_PPP 71 (art. 6)` | Sem `sem marcas` ou `para diagramar` em DOCX |
| PPP 71 - artigo 7 | `OneDrive_2_24-06-2026/236012_Revista_PPP 71 (art. 7)` | Sem `sem marcas` ou `para diagramar` em DOCX |
| TD 3181 - Expansão de área | `OneDrive_2_24-06-2026/236222_TD 3181_Expansão de área` | Sem `sem marcas` ou `para diagramar` em DOCX |
| PPP 72 - artigo 1 | `OneDrive_2_24-06-2026/236699_Revista_PPP 72 (art. 1)` | Sem `sem marcas` ou `para diagramar` em DOCX |
| PPP 72 - artigo 4 | `OneDrive_2_24-06-2026/236699_Revista_PPP 72 (art. 4)` | Sem `sem marcas` ou `para diagramar` em DOCX |
| TD 3202 - Carbono zero | `OneDrive_2_24-06-2026/236767_TD 3202_Carbono zero` | Sem `sem marcas` ou `para diagramar` em DOCX |

## Itens que não são executáveis como teste

| Documento | Motivo |
|---|---|
| Livro Monitoramento e avaliação - capítulo 3 | Não possui `original.docx` |
| PPE 55(2) - artigo 5 | Não possui `original.docx` |

## Próximo passo para obter um holdout de 10

Para formar um conjunto automático de dez documentos sem contaminação, é necessário recuperar a versão final em DOCX de pelo menos um dos nove candidatos acima e adicionar outro documento nunca usado no aprendizado. Alternativamente, os nove candidatos atuais podem ser usados já para avaliação qualitativa e manual contra os PDFs finais.
