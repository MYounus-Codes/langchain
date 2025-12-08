# LangChain Course — Agentic AI Projects

A comprehensive, example-driven learning resource for building intelligent agents and applications using LangChain. This repository provides modular components, practical examples, and best practices for working with Language Models, chains, prompts, memory systems, and advanced AI patterns.

**Status:** Active Development — Comprehensive learning repository with expanding examples and documentation.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [What You'll Learn](#what-youll-learn)
3. [Quick Start](#quick-start)
4. [Project Structure](#project-structure)
5. [Core Components](#core-components)
6. [LangChain Fundamentals](#langchain-fundamentals)
7. [Model Integrations](#model-integrations)
8. [Prompting Techniques](#prompting-techniques)
9. [Running Examples](#running-examples)
10. [Development Guide](#development-guide)
11. [Best Practices](#best-practices)
12. [Troubleshooting](#troubleshooting)
13. [Contributing](#contributing)
14. [Resources](#resources)

---

## Overview

This repository is a structured learning platform for mastering LangChain, an open-source framework for building applications powered by language models. Whether you're a beginner learning LLM fundamentals or an advanced developer building production systems, this course provides practical, working examples organized by concept.

### Key Features

- **Modular Architecture**: Components organized by functionality (agents, chains, prompts, memory, models)
- **Progressive Learning**: From fundamentals to advanced patterns
- **Multiple LLM Integrations**: OpenAI, Google Gemini, Anthropic Claude, Hugging Face
- **Real-world Examples**: Chatbots, semantic search, document processing
- **Best Practices**: Security, performance, and production-ready patterns
- **Active Development**: Continuously updated with latest LangChain features

### Goals

- Provide runnable, minimal examples showing end-to-end workflows
- Keep components small, focused, and remixable
- Serve as a learning resource for LLM application development
- Demonstrate industry best practices and patterns
- Enable rapid prototyping and experimentation

---

## What You'll Learn

### Beginner Level
- ✅ Understanding Language Models and their capabilities
- ✅ Setting up LangChain and basic configuration
- ✅ Creating your first LLM chain
- ✅ Working with different model providers
- ✅ Basic prompt engineering

### Intermediate Level
- ✅ Advanced prompting techniques (few-shot, chain-of-thought, role-based)
- ✅ Building conversational AI with memory
- ✅ Creating custom chains and agents
- ✅ Embeddings and semantic search
- ✅ Document processing and indexing

### Advanced Level
- ✅ Building autonomous agents with tools
- ✅ Complex memory management strategies
- ✅ Production deployment patterns
- ✅ Performance optimization
- ✅ Error handling and resilience

---

## Quick Start

### Prerequisites

- **Python 3.9+** (recommend 3.10 or 3.11)
- **pip** or **conda** for package management
- **API Keys** for LLM providers (optional, depends on which models you use):
  - OpenAI API key (for GPT models)
  - Google Cloud credentials (for Gemini)
  - Anthropic API key (for Claude)
  - Hugging Face API key (for HF models)

### Installation Steps

#### 1. Clone and Navigate to Repository
```powershell
cd d:\agentic-ai-projects\langchain_course
```

#### 2. Create Virtual Environment
```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# If you get execution policy error, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 3. Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Configure Environment Variables
```powershell
# Create a .env file in project root
New-Item .env

# Add your API keys (do NOT commit this file)
# OPENAI_API_KEY=your_key_here
# GOOGLE_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here
```

#### 5. Run Your First Example
```powershell
python main.py
```

---

## Project Structure

```
langchain_course/
│
├── main.py                          # Entry point - run this first
├── google_sdk.py                    # Google SDK configuration
├── pyproject.toml                   # Project metadata and dependencies
├── README.md                        # This file
│
├── fundamentals/                    # Beginner-level learning materials
│   └── langchain_helloworld.py      # Minimal LangChain example
│
├── huggingface_api_models/          # Hugging Face model integration
│   └── 1_chatmodel_hf_api.py        # Chat model via HF API
│
└── components/                      # Core modular components
    ├── agents/                      # Agent implementations
    │   └── __init__.py
    ├── chains/                      # Reusable chain definitions
    │   └── __init__.py
    ├── indexes/                     # Vector stores and retrieval
    │   └── __init__.py
    ├── memory/                      # Memory management systems
    │   └── __init__.py
    ├── models/                      # LLM model integrations
    │   ├── chatmodels/              # Chat-based models
    │   │   ├── 1_chatmodel_gemini.py
    │   │   ├── 2_chatmodel_openai.py
    │   │   └── 3_chatmodel_claude.py
    │   ├── embeddedmodels/          # Embedding models
    │   │   ├── 1_embeddings_gemini_query.py
    │   │   ├── 2_embeddings_gemini_docs.py
    │   │   └── 3_embeddings_hf_local.py
    │   ├── llms/                    # Standard LLM models
    │   │   └── 1_llm_demo.py
    │   └── class_projects/          # Practical projects
    │       ├── 1_document_similarity.py
    │       ├── 2_document_similarity_hf_local.py
    │       ├── simple_chatbot.py
    │       └── what_is_project.txt
    └── prompts/                     # Prompt templates and techniques
        ├── what_are_prompts.txt     # Comprehensive prompting guide
        └── class_projects/
```

### Directory Descriptions

| Directory | Purpose |
|-----------|---------|
| `fundamentals/` | Entry point with basic LangChain examples |
| `huggingface_api_models/` | Examples using Hugging Face API |
| `components/agents/` | Building autonomous agents with tools |
| `components/chains/` | Creating multi-step processing chains |
| `components/indexes/` | Vector databases and retrieval systems |
| `components/memory/` | Conversation memory and state management |
| `components/models/` | Integration with various LLM providers |
| `components/prompts/` | Prompt templates and engineering techniques |

---

## Core Components

### 1. Agents (components/agents/)
Agents are systems that can plan, reason, and take actions using tools.

**Key Concepts:**
- Tool use and function calling
- Reasoning and planning
- Error handling and retries
- Multi-step task execution

**Example Use Cases:**
- Question answering systems
- Automated data analysis
- Web scraping and API integration
- Report generation

### 2. Chains (components/chains/)
Chains are sequences of calls to language models and other tools.

**Key Concepts:**
- Composition of multiple steps
- Data transformation pipelines
- Error handling between steps
- Chain templates and reusability

**Example Use Cases:**
- Document summarization
- Information extraction
- Multi-stage transformations
- Workflow automation

### 3. Memory (components/memory/)
Memory systems allow applications to maintain context across conversations.

**Types of Memory:**
- **Buffer Memory**: Simple conversation history
- **Summary Memory**: Condensed conversation summaries
- **Managed Memory**: Vector store-based semantic memory
- **Entity Memory**: Tracking specific entities across conversations

### 4. Prompts (components/prompts/)
Prompt engineering is the art of crafting effective instructions for LLMs.

**Techniques Covered:**
- Static vs Dynamic Prompts
- Few-shot prompting with examples
- Chain-of-thought reasoning prompts
- Role-based prompting
- Structured output prompting
- Prompt templates and composition

See `components/prompts/what_are_prompts.txt` for comprehensive guide.

### 5. Models (components/models/)

#### Chat Models (components/models/chatmodels/)
Conversational interfaces with multiple turns.

**Supported Providers:**
- **OpenAI**: GPT-3.5, GPT-4
- **Google**: Gemini Pro
- **Anthropic**: Claude
- **Hugging Face**: Open-source models

#### Embedding Models (components/models/embeddedmodels/)
Convert text to high-dimensional vectors for semantic understanding.

**Use Cases:**
- Semantic search and similarity
- Vector database indexing
- Document clustering
- Recommendation systems

**Supported Providers:**
- Google Embeddings API
- Hugging Face Transformers (local)
- OpenAI Embeddings

#### LLM Models (components/models/llms/)
Standard language model interfaces for completion-based tasks.

### 6. Indexes (components/indexes/)
Vector databases and retrieval augmented generation (RAG) systems.

**Components:**
- Document loaders and processors
- Vector stores and embeddings
- Retrieval chains
- Semantic search

---

## LangChain Fundamentals

### What is LangChain?

LangChain is a framework for developing applications powered by language models. It enables you to:
- Connect language models to various data sources
- Allow language models to interact with their environment
- Build complex applications with multiple components
- Deploy production-ready AI systems

### Core Concepts

#### 1. **Language Models (LMs)**
- **LLMs**: Large Language Models for text completion
- **Chat Models**: Optimized for conversation with system/human/assistant messages
- **Embeddings**: Convert text to numerical vectors

#### 2. **Prompts**
Instructions sent to language models. Can be:
- **Static**: Fixed prompts for consistent behavior
- **Dynamic**: Variable prompts adapting to input

#### 3. **Chains**
Sequences of calls to language models and other tools:
```
Input → Prompt → LLM → Output Parser → Result
```

#### 4. **Memory**
Persistent storage of conversation history and context:
```
Current Turn → Memory → Context → LLM Response
```

#### 5. **Agents**
Systems with access to tools that can reason and plan:
```
User Query → Agent → Tool 1, Tool 2, ... → Response
```

#### 6. **Output Parsers**
Convert raw LLM outputs to structured formats:
```
Raw Text → Parser → Structured Data (JSON, CSV, etc.)
```

---

## Model Integrations

### OpenAI Models

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4", temperature=0.7)
response = model.invoke("Hello, how are you?")
```

**Available Models:**
- `gpt-4`: Most capable model
- `gpt-3.5-turbo`: Fast and cost-effective
- `text-embedding-3-large`: State-of-the-art embeddings

### Google Gemini

```python
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-pro")
response = model.invoke("Explain quantum computing")
```

### Anthropic Claude

```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-3-opus-20240229")
response = model.invoke("What is machine learning?")
```

### Hugging Face Models (Local)

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embedded = embeddings.embed_query("Hello world")
```

---

## Prompting Techniques

### 1. Basic Prompt Template
```python
from langchain_core.prompts import PromptTemplate

template = "What is the capital of {country}?"
prompt = PromptTemplate(
    template=template,
    input_variables=["country"]
)
```

### 2. Chat Prompt Template
```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
])
```

### 3. Few-Shot Prompting
```python
from langchain_core.prompts import FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "big", "output": "small"},
]

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=PromptTemplate(...),
    suffix="Input: {word}\nOutput:",
    input_variables=["word"]
)
```

### 4. Chain-of-Thought Prompting
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Think step by step before answering."),
    ("human", "{problem}"),
])
```

See `components/prompts/what_are_prompts.txt` for comprehensive prompting guide with 11 advanced techniques.

---

## Running Examples

### Example 1: Basic LLM Call
```powershell
python components/models/llms/1_llm_demo.py
```

### Example 2: Chat Models
```powershell
python components/models/chatmodels/1_chatmodel_gemini.py
python components/models/chatmodels/2_chatmodel_openai.py
python components/models/chatmodels/3_chatmodel_claude.py
```

### Example 3: Embeddings and Semantic Search
```powershell
python components/models/embeddedmodels/1_embeddings_gemini_query.py
python components/models/embeddedmodels/2_embeddings_gemini_docs.py
python components/models/embeddedmodels/3_embeddings_hf_local.py
```

### Example 4: Document Similarity
```powershell
python components/models/class_projects/1_document_similarity.py
python components/models/class_projects/2_document_similarity_hf_local.py
```

### Example 5: Simple Chatbot
```powershell
streamlit run components/models/class_projects/simple_chatbot.py
```

### Example 6: Fundamentals
```powershell
python fundamentals/langchain_helloworld.py
```

---

## Development Guide

### Setting Up Your Development Environment

```powershell
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install development tools (optional)
pip install pytest black flake8 mypy
```

### Creating New Components

#### Structure for New Component:
```python
# components/new_component/my_feature.py

"""
Module: my_feature
Description: Brief description of what this does
Author: Your Name
Date: YYYY-MM-DD
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def main():
    """Main function demonstrating the feature."""
    # Your implementation here
    pass

if __name__ == "__main__":
    main()
```

#### Best Practices for Components:
1. **Keep it focused**: One feature per file
2. **Add docstrings**: Explain what the code does
3. **Use type hints**: For better IDE support
4. **Handle errors gracefully**: Try-except blocks
5. **Configuration**: Use environment variables for secrets
6. **Documentation**: Add comments for complex logic

### Adding Dependencies

1. Update `requirements.txt`:
```
langchain>=0.1.0
langchain-openai>=0.1.0
langchain-google-genai>=0.1.0
```

2. Or use `pyproject.toml` for more advanced configuration

3. Install new dependencies:
```powershell
pip install -r requirements.txt
```

### Testing Your Code

```powershell
# Run all tests
pytest

# Run specific test file
pytest tests/test_my_feature.py

# Run with coverage
pytest --cov=components tests/
```

---

## Best Practices

### 1. **Security**
- ✅ Never hardcode API keys
- ✅ Use environment variables (.env files)
- ✅ Don't commit .env files to git
- ✅ Validate user inputs to prevent prompt injection
- ✅ Use `.gitignore` to exclude sensitive files

### 2. **Performance**
- ✅ Cache embeddings and responses
- ✅ Use appropriate model sizes for your use case
- ✅ Implement rate limiting
- ✅ Monitor token usage
- ✅ Use streaming for long responses

### 3. **Cost Management**
- ✅ Monitor API usage and costs
- ✅ Use cheaper models for simple tasks
- ✅ Implement caching to avoid redundant calls
- ✅ Batch requests when possible
- ✅ Set token limits for responses

### 4. **Error Handling**
- ✅ Implement retry logic with exponential backoff
- ✅ Handle rate limiting gracefully
- ✅ Provide meaningful error messages
- ✅ Log errors for debugging
- ✅ Fallback mechanisms for failures

### 5. **Code Quality**
- ✅ Use type hints
- ✅ Write clear docstrings
- ✅ Follow PEP 8 style guide
- ✅ Keep functions small and focused
- ✅ Use meaningful variable names

### 6. **Production Readiness**
- ✅ Comprehensive error handling
- ✅ Logging and monitoring
- ✅ Configuration management
- ✅ Test coverage (>80%)
- ✅ Documentation

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: "ModuleNotFoundError: No module named 'langchain'"
**Solution:**
```powershell
pip install langchain langchain-core
```

#### Issue 2: "AuthenticationError: Invalid API key"
**Solution:**
- Check `.env` file has correct API key
- Verify environment variable is loaded
- Test API key validity on provider's website

#### Issue 3: "RateLimitError: Too many requests"
**Solution:**
```python
import time

def call_with_retry(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError:
            wait_time = 2 ** attempt  # Exponential backoff
            time.sleep(wait_time)
    raise Exception("Max retries exceeded")
```

#### Issue 4: "Execution Policy" error on Windows
**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Issue 5: Memory issues with large documents
**Solution:**
- Process documents in chunks
- Use streaming instead of loading entire documents
- Implement document pagination

### Debugging Tips

1. **Enable debug logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Print intermediate results:**
```python
print(f"Input: {input_data}")
print(f"Processed: {processed_data}")
print(f"Output: {final_output}")
```

3. **Use Python debugger:**
```python
import pdb; pdb.set_trace()
```

---

## Contributing

We welcome contributions! Here's how to contribute:

### Steps to Contribute:
1. **Fork the repository** on GitHub
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Make your changes** and add tests
4. **Follow code style**: Use `black` and `flake8`
5. **Commit with clear messages**: `git commit -m "Add feature description"`
6. **Push to your fork**: `git push origin feature/my-feature`
7. **Create Pull Request** with description of changes

### Code Style Guidelines:
- Use **black** for formatting
- Use **flake8** for linting
- Write **type hints** for functions
- Add **docstrings** to all functions and classes
- Include **comments** for complex logic

### Testing Requirements:
- Write tests for new features
- Ensure all tests pass: `pytest`
- Aim for >80% code coverage
- Include both unit and integration tests

---

## Resources

### Official Documentation
- [LangChain Documentation](https://python.langchain.com/)
- [LangChain GitHub Repository](https://github.com/langchain-ai/langchain)
- [LangChain API Reference](https://api.python.langchain.com/)

### Model Provider Documentation
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Google Gemini Docs](https://ai.google.dev/)
- [Anthropic Claude Docs](https://docs.anthropic.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

### Learning Resources
- [LangChain by Example](https://python.langchain.com/docs/get_started/introduction.html)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [RAG Pattern](https://python.langchain.com/docs/use_cases/question_answering/)

### Recommended Courses
- **LangChain for LLM Application Development** - DeepLearning.AI
- **Building Systems with the ChatGPT API** - OpenAI/DeepLearning.AI
- **Advanced Retrieval-Augmented Generation** - Various platforms

### Community
- [LangChain Discord Community](https://discord.gg/langchain)
- [LangChain GitHub Discussions](https://github.com/langchain-ai/langchain/discussions)
- [Stack Overflow - LangChain Tag](https://stackoverflow.com/questions/tagged/langchain)

---

## Project Statistics

| Metric | Value |
|--------|-------|
| **Python Version** | 3.9+ |
| **LangChain Version** | 0.1.0+ |
| **Total Components** | 6 major (Agents, Chains, Memory, Prompts, Models, Indexes) |
| **Model Integrations** | 5+ providers |
| **Example Projects** | 5+ |
| **Lines of Code** | Growing |

---

## Roadmap

### Completed ✅
- Basic LangChain fundamentals
- Multi-provider model integrations
- Prompt engineering guide
- Document similarity examples
- Simple chatbot example

### In Progress 🔄
- Advanced agent patterns
- Production deployment guides
- Performance optimization tutorials
- Comprehensive testing suite

### Planned 📋
- CI/CD pipeline setup
- Docker containerization
- API server example
- Advanced memory strategies
- Multi-agent collaboration patterns

---

## License & Contact

**License:** MIT License  
**Author:** MYounus-Codes  
**Repository:** [GitHub - langchain_course](https://github.com/MYounus-Codes/langchain)  

For questions, issues, or suggestions:
1. Open an issue on GitHub
2. Contact the maintainers
3. Join the LangChain community

---

## Quick Reference

### Essential Commands

```powershell
# Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run examples
python main.py
python fundamentals/langchain_helloworld.py

# Run chatbot
streamlit run components/models/class_projects/simple_chatbot.py

# Code quality
black components/
flake8 components/
pytest

# View documentation
python -c "import langchain; help(langchain)"
```

### Key Files Quick Reference

| File | Purpose |
|------|---------|
| `main.py` | Main entry point |
| `components/prompts/what_are_prompts.txt` | Comprehensive prompting guide |
| `fundamentals/langchain_helloworld.py` | Beginner tutorial |
| `components/models/class_projects/simple_chatbot.py` | Interactive chatbot demo |
| `.env` | API keys (create this file) |
| `pyproject.toml` | Project configuration |

---

**Happy Learning! 🚀**

Start with `fundamentals/langchain_helloworld.py` if you're new to LangChain, or explore the specific components that interest you.

# Run a demo
python main.py
```

## Contributing

Contributions are welcome. Good first contributions:
- Add small, well-documented example scripts.
- Improve README sections with explicit instructions for each demo.
- Add automated tests for components where feasible.

When contributing:
- Fork the repo and open a pull request against `main`.
- Describe the change and which demo or example it affects.

## Next Steps & Suggestions

- Add a `requirements.txt` or fully populate `pyproject.toml` with dev dependencies.
- Add CI (GitHub Actions) for linting and tests.
- Provide environment example files like `.env.example` showing required vars.
- Add badges for build status, Python version, and license.

## License

Specify your project's license here (e.g., MIT). If you want MIT, add a `LICENSE` file with the MIT text and replace this section with:

- **License:** MIT — see `LICENSE` file.

## Contact

If you have questions or want to collaborate, open an issue or reach out via the GitHub repo: `MYounus-Codes/langchain`.

---

If you want, I can also:
- generate a `requirements.txt` with common packages used by LangChain demos,
- add a `.env.example` and a small sample GitHub Actions CI workflow,
- or populate `main.py` with a simple runnable demo that uses `components/llms/1_llm_demo.py`.

