# Agentes editoriais

- `sinopse_abstract` (`sin`): SINOPSE, ABSTRACT, palavras-chave, keywords e JEL.
- `gramatica_ortografia` (`gram`): erros linguísticos objetivos e microerros mecânicos; sem reescrita estilística.
- `tabelas_figuras` (`tab`): identificação, título, subtítulo, fonte, elaboração, unidade e notas com evidência.
- `estrutura` (`est`): títulos reais, hierarquia e numeração; ignora pré-textuais e falsos títulos.
- `comentarios_usuario_referencias`: atende pedido explícito para localizar ou incluir referência.
- `referencias` (`ref`): corpo–lista e ABNT NBR 6023:2025/NBR 10520:2023; nunca inventa dados.
- `tipografia` (`tip`): atributos visuais; não altera conteúdo.
- `coordenador` (`coord`): consolida comentários aceitos.

O perfil `TD` pode ser detectado por nomes como `123456_TD_2345.docx`.

Aceitar somente evidência local, fragmento mínimo, correção concreta e escopo correto. Validar e deduplicar após o merge.

