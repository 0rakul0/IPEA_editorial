# Agentes editoriais

- `sinopse_abstract` (`sin`): SINOPSE, ABSTRACT, palavras-chave, keywords e JEL. Aponta apenas inconsistências objetivas.
- `gramatica_ortografia` (`gram`): ortografia, acentuação, concordância, regência, crase, pontuação obrigatória e microerros mecânicos. Não reescreve por elegância.
- `tabelas_figuras` (`tab`): identificação, título, subtítulo, fonte, elaboração, unidade e notas quando o bloco fornece evidência.
- `estrutura` (`est`): títulos e subtítulos reais, hierarquia e numeração. Ignora pré-textuais, citações, listas, tabelas, legendas e referências.
- `comentarios_usuario_referencias`: atende comentários explícitos do editor pedindo busca ou inclusão de referência; usa candidatos fornecidos e evita duplicação.
- `referencias` (`ref`): lista bibliográfica e coerência corpo–lista, com ABNT NBR 6023:2025 e NBR 10520:2023. Não inventa elementos.
- `tipografia` (`tip`): tamanho, caixa, negrito, itálico, alinhamento, recuo, espaçamento e entrelinha. Não altera conteúdo.
- `coordenador` (`coord`): consolida os comentários aceitos.

## Perfil TD

O nome do arquivo pode ativar `TD` quando segue padrão semelhante a `123456_TD_2345.docx`.

## Aceitação

- Evidência local e objetiva.
- `issue_excerpt` mínimo e ancorável.
- `suggested_fix` concreto e materialmente diferente.
- Um problema por item.
- Rejeição de comentários vagos, especulativos, redundantes ou fora de escopo.
- Deduplicação após o merge.

Os agentes trabalham de forma independente. Os lotes preservam memória local do agente; o merge ocorre ao final.

