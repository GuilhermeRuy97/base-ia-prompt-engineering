# Curso de Prompt Engineering

Este repositório contém os exercícios práticos e exemplos da disciplina de Prompt Engineering do MBA em Engenharia de Software com IA.

## Estrutura dos capítulos

### 📁 1-tipos-de-prompts
Fundamentos de prompt engineering com 9 técnicas essenciais:
- Role-based prompting
- Zero-shot e Few-shot learning
- Chain of Thought (CoT) e variações
- Tree of Thoughts (ToT)
- Skeleton of Thought (SoT)
- ReAct framework
- Prompt chaining
- Least-to-most decomposition

### 📁 4-prompts-e-workflow-de-agentes
Implementações de workflows baseados em agentes para:
- Análise arquitetural de código
- Auditoria de dependências
- Orquestração de comandos entre agentes

### 📁 5-gerenciamento-e-versionamento-de-prompts
Sistema avançado de gerenciamento de prompts com:
- Versionamento local usando YAML
- Integração com LangSmith para colaboração
- Agentes especializados para code review e criação de PRs
- Testes automatizados com pytest

### 📁 6-prompt-enriquecido
Técnicas avançadas de enriquecimento de prompts:
- Query expansion
- ITER-RETGEN (Iterative Retrieval Generation)
- Enriquecimento contextual de queries

## Configuração do Ambiente

**Importante:** Cada pasta do curso possui seu próprio ambiente virtual e dependências específicas.

### 1. Criar e Ativar Ambiente Virtual

```bash
# Navegue até a pasta desejada
cd [pasta-do-capítulo]

# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
# No macOS/Linux:
source venv/bin/activate

# No Windows:
venv\Scripts\activate
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 3. Configuração das Variáveis de Ambiente

```bash
# Copiar arquivo de exemplo
cp .env.example .env

# Editar o arquivo .env e adicionar suas chaves
# Minimamente necessário: OPENAI_API_KEY=sua_chave_aqui
```
## Dependências Principais

As dependências variam entre os capítulos:

- **Capítulos 1 e 4:** LangChain 0.3.x (versão estável)
- **Capítulos 5 e 6:** LangChain 1.0.0a5 com LangGraph para recursos avançados

Para detalhes específicos de cada capítulo, consulte o arquivo `requirements.txt` correspondente.