# 🔗 Langchain-Chains

This repository captures my experiments with **Langchain chains** — exploring how to use sequential, parallel, and conditional workflows for different use cases with HuggingFace-hosted LLMs like **Mistral**. It also features usage of `StrOutputParser` and `PydanticOutputParser` for handling outputs intelligently.

---

## 🧠 What I Explored & Learnt

- **Runnable Chains**  
  Built multiple chain types using:
  - `RunnableSequence` for step-by-step executioscacsabciaccsacac
  - `RunnableParallel` for concurrent processing
  - `RunnableBranch` for condition-based execution

- **Prompt Engineering**  
  Practiced crafting templates for summarisation, Q&A generation, and feedback response.

- **Output Parsers**  
  - `StrOutputParser` for basic string outputs  
  - `PydanticOutputParser` for structured parsing using Pydantic models

- **Multi-Model Chaining**  
  Used both **Mistral** and **Llama 3** in a single flow for specific tasks.

- **Sentiment-Based Conditional Flow**  
  Used branching logic to handle positive vs negative feedback differently.

---

## 📂 Key Files

- `simple_chains.py` – Shows a basic linear prompt → model → parser setup
- `sequential_chain.py` – Multi-step LLM chain with intermediate outputs
- `parallel-chains.py` – Uses two different LLMs in parallel for generating notes and quizzes
- `conditional_chain.py` – Classifies sentiment and routes response accordingly using `RunnableBranch`
- `test.py` – Used to verify HuggingFace Langchain integration is working

---

## 🤝 Author

**Nipun Kumar**  
[Portfolio](https://nipun.framer.website/) • [GitHub](https://github.com/nipunraj96) • [LinkedIn](https://www.linkedin.com/in/nipunkumar01)
