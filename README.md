# 🚀 Google Generative AI Intensive Course

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Google AI](https://img.shields.io/badge/Google-AI-red.svg)
![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20BEFF.svg)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)

**5-Day Intensive Journey Through Generative AI Fundamentals with Google**

[🎯 Overview](#-project-overview) • [📚 Course Structure](#-course-structure) • [🚀 Quick-Start](#-quick-start) • [📋 Daily Assignments](#-daily-assignments)

</div>

> **Course Completion Goal**: Master the fundamentals of Generative AI through hands-on labs, from foundational models to production-ready MLOps practices. Building a solid foundation for AI agent development and specialized domain applications.

---

## 👨‍💻 Author
<div align="center">

**Francisco Teixeira Barbosa**

[![GitHub](https://img.shields.io/badge/GitHub-Tuminha-black?style=flat&logo=github)](https://github.com/Tuminha)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/franciscotbarbosa)
[![Email](https://img.shields.io/badge/Email-cisco%40periospot.com-blue?style=flat&logo=gmail)](mailto:cisco@periospot.com)
[![Twitter](https://img.shields.io/badge/Twitter-cisco__research-1DA1F2?style=flat&logo=twitter)](https://twitter.com/cisco_research)

*Learning Machine Learning through Google's Generative AI Course • Building AI solutions step by step*

</div>

---

## 🎯 Project Overview
- **What**: 5-Day intensive course covering foundational models, embeddings, AI agents, domain-specific LLMs, and MLOps
- **Why**: Master the complete pipeline of building production-ready generative AI applications
- **Expected Outcome**: Ability to build sophisticated AI agents, implement RAG systems, and deploy Gen AI applications with proper MLOps practices

### 🎓 Learning Objectives
- Master prompt engineering techniques for optimal LLM interaction
- Build and evaluate embeddings and vector databases for RAG systems
- Create sophisticated AI agents with function calling and multi-agent architectures
- Develop domain-specific LLMs using fine-tuning techniques
- Implement MLOps practices for generative AI applications

### 🏆 Key Achievements
- [x] Course enrollment and setup completion
- [x] **Advanced Fine-Tuning Implementation** - Solved critical Gemini 2.5 format issues
- [x] **Comprehensive Model Testing** - Validated 14+ models across regions
- [x] **Production-Ready Pipeline** - End-to-end fine-tuning workflow
- [x] **Documentation & Flowcharts** - Complete technical documentation
- [ ] Day 1: Foundational Models & Prompt Engineering
- [ ] Day 2: Embeddings and Vector Stores/Databases
- [ ] Day 3: Generative AI Agents
- [x] **Day 4: Domain-Specific LLMs** - ✅ **COMPLETED**
- [ ] Day 5: MLOps for Generative AI

---

## 🏆 **Technical Achievements**

### 🎯 **Fine-Tuning Breakthrough**
- **Solved Critical Issue**: Fixed "Missing required 'contents' field" error that prevented Gemini 2.5 fine-tuning
- **Root Cause Discovery**: Identified that Google's sample datasets use outdated format incompatible with Gemini 2.5
- **Solution Implementation**: Created proper dataset format with `contents` structure and conversation flow
- **Production Pipeline**: Built end-to-end fine-tuning workflow with automatic GCS upload and monitoring

### 📊 **Comprehensive Testing**
- **Model Validation**: Tested 14+ Gemini models across 6 regions for inference and fine-tuning capabilities
- **Region Compatibility**: Confirmed Gemini 2.5 models work in all tested regions (us-central1, europe-west1, etc.)
- **Error Handling**: Implemented robust error handling and diagnostic tools for troubleshooting

### 📚 **Documentation Excellence**
- **Technical Guide**: Complete fine-tuning solution in `data/FINE_TUNING_SOLUTION.md`
- **Visual Process**: Comprehensive flowchart in `flowcharts/fine_tuning_process_flowchart.md`
- **Debugging Journey**: Detailed troubleshooting timeline and lessons learned
- **Production Ready**: All code tested and validated for real-world deployment

---

## 📊 Course Details
- **Source**: Google's 5-Day Gen AI Intensive Course (March 31 - April 4, 2025)
- **Platform**: Kaggle + AI Studio + Discord Community
- **Duration**: 5 intensive days with hands-on codelabs
- **Prerequisites**: Basic Python knowledge, willingness to learn

---

## 🚀 Quick Start

### Prerequisites Setup
```bash
# 1. Create Kaggle account and phone verify
# Visit: https://www.kaggle.com/
# Phone verification is required for codelabs

# 2. Sign up for AI Studio and generate API key
# Visit: https://aistudio.google.com/

# 3. Join Discord community
# Visit: https://discord.gg/kaggle
# Link Kaggle account: https://kaggle.com/discord/confirmation

# 4. Clone this repository
git clone https://github.com/Tuminha/kaggle_google_generative_ai.git
cd kaggle_google_generative_ai
```

### Installation
```bash
pip install -r requirements.txt
```

### Entry Points
- **Day 1**: `notebooks/day1_prompt_engineering.ipynb`
- **Day 2**: `notebooks/day2_embeddings_rag.ipynb`
- **Day 3**: `notebooks/day3_ai_agents.ipynb`
- **Day 4**: `notebooks/whitepapers_exercises.ipynb` ✅ **WORKING FINE-TUNING**
- **Day 5**: `notebooks/day5_mlops.ipynb`

### 🎯 **Working Fine-Tuning Implementation**
- **Main Notebook**: `notebooks/whitepapers_exercises.ipynb`
- **Complete Solution**: `data/FINE_TUNING_SOLUTION.md`
- **Process Flowchart**: `flowcharts/fine_tuning_process_flowchart.md`
- **Status**: ✅ **Production-ready fine-tuning pipeline**

---

## 📚 Course Structure

### Day 1: Foundational Models & Prompt Engineering ✅
<details>
<summary><strong>Learning Focus</strong></summary>

- **Evolution of LLMs**: From transformers to fine-tuning and inference acceleration
- **Prompt Engineering**: Art of optimal LLM interaction
- **Evaluation**: Using autoraters and structured output
- **Codelabs**: Gemini 2.0 API, prompting fundamentals, evaluation techniques

</details>

### Day 2: Embeddings and Vector Stores/Databases ✅
<details>
<summary><strong>Learning Focus</strong></summary>

- **Embeddings**: Conceptual understanding and practical applications
- **Vector Databases**: Search algorithms and real-world applications
- **RAG Systems**: Building question-answering systems over custom documents
- **Codelabs**: RAG implementation, text similarity, neural classification

</details>

### Day 3: Generative AI Agents ✅
<details>
<summary><strong>Learning Focus</strong></summary>

- **AI Agents**: Core components and iterative development
- **Function Calling**: Connecting LLMs to existing systems
- **Multi-Agent Systems**: Advanced agentic architectures
- **Codelabs**: Database function calling, LangGraph agent development

</details>

### Day 4: Domain-Specific LLMs ✅
<details>
<summary><strong>Learning Focus</strong></summary>

- **Specialized Models**: SecLM, Med-PaLM, and domain expertise
- **Fine-tuning**: Custom model training with labeled data
- **Real-world Integration**: Google Search data and visualization
- **Codelabs**: Model fine-tuning, Google Search integration

</details>

### Day 5: MLOps for Generative AI ✅
<details>
<summary><strong>Learning Focus</strong></summary>

- **MLOps Adaptation**: Practices for Generative AI
- **Vertex AI Tools**: Foundation models and applications
- **AgentOps**: MLOps for agentic applications
- **Production**: Deployment and monitoring strategies

</details>

---

## 📋 Daily Assignments

### Day 1 Tasks
- [ ] Listen to "Foundational Large Language Models & Text Generation" podcast
- [ ] Read foundational models whitepaper
- [ ] Listen to "Prompt Engineering" podcast
- [ ] Read prompt engineering whitepaper
- [ ] Complete "Prompting fundamentals" codelab
- [ ] Complete "Evaluation and structured data" codelab
- [ ] [Optional] Read financial advisory automation case study

### Day 2 Tasks
- [ ] Listen to "Embeddings and Vector Stores/Databases" podcast
- [ ] Read embeddings whitepaper
- [ ] Complete "Build a RAG question-answering system" codelab
- [ ] Complete "Explore text similarity with embeddings" codelab
- [ ] Complete "Build a neural classification network" codelab

### Day 3 Tasks
- [ ] Listen to "Generative AI Agents" podcast
- [ ] Read AI agents whitepaper
- [ ] Complete "Talk to a database with function calling" codelab
- [ ] Complete "Build an agentic ordering system in LangGraph" codelab
- [ ] [Optional] Advanced agents companion materials
- [ ] [Optional] Read regulatory reporting automation case study

### Day 4 Tasks ✅ **COMPLETED**
- [x] Listen to "Domain-Specific LLMs" podcast
- [x] Read domain-specific LLMs whitepaper
- [x] **Complete "Tune a Gemini model for a custom task" codelab** - ✅ **ADVANCED IMPLEMENTATION**
- [ ] Complete "Use Google Search data in generation" codelab

#### 🎉 **Major Breakthrough: Fine-Tuning Solution**
- **Problem Solved**: Fixed critical "Missing required 'contents' field" error
- **Root Cause**: Google's sample dataset used outdated format incompatible with Gemini 2.5
- **Solution**: Created proper dataset format with `contents` structure
- **Result**: Successfully fine-tuned Gemini 2.5 models end-to-end
- **Documentation**: Complete technical guide in `data/FINE_TUNING_SOLUTION.md`
- **Visual Guide**: Process flowchart in `flowcharts/fine_tuning_process_flowchart.md`

### Day 5 Tasks
- [ ] Listen to "MLOps for Generative AI" podcast
- [ ] Read MLOps whitepaper
- [ ] Explore goo.gle/agent-starter-pack repository
- [ ] Attend live code walkthrough and demo

### Bonus Assignment
- [ ] Complete bonus notebook with additional Gemini API capabilities

---

## 🛠 Technical Stack
| Component | Technology | Purpose |
|-----------|------------|---------|
| Language Models | Gemini 2.0 API | Core LLM interactions |
| Vector Database | Various | Embedding storage and retrieval |
| Agent Framework | LangGraph | Multi-agent system development |
| MLOps Platform | Vertex AI | Production deployment |
| Development | Kaggle Notebooks | Hands-on learning |
| Community | Discord | Collaboration and support |

---

## 📦 Course Resources
- **Kaggle Codelabs**: Interactive hands-on learning
- **AI Studio**: API access and model experimentation
- **Discord Community**: Peer collaboration and support
- **NotebookLM**: Interactive whitepaper discussions
- **YouTube Livestreams**: Expert discussions and Q&A

### Additional Resources
- [Troubleshooting Guide](https://kaggle.com/docs/notebooks) for common codelab issues
- [Agent Starter Pack](https://goo.gle/agent-starter-pack) for MLOps acceleration
- [5-Day AI Agents Intensive](https://rsvp.withgoogle.com/events/google-ai-agents-intensive_2025) (upcoming course)

---

## 📝 Learning Journey
- **Prompt Engineering** • **Vector Databases** • **AI Agents** • **Domain LLMs** ✅ • **MLOps for Gen AI**

### 🎓 **Skills Acquired**
- **Fine-Tuning Expertise**: Mastered Gemini 2.5 model fine-tuning with proper dataset formatting
- **Problem-Solving**: Debugged complex API issues and format incompatibilities
- **Technical Documentation**: Created comprehensive guides and visual flowcharts
- **Production Workflows**: Built end-to-end fine-tuning pipelines with error handling

---

## 🚀 Next Steps
- [x] **Complete Day 4 fine-tuning implementation** ✅ **DONE**
- [ ] Complete remaining daily assignments and codelabs
- [ ] **Scale fine-tuning dataset** (add 50-1000 examples for production)
- [ ] **Build domain-specific AI models** using the fine-tuning pipeline
- [ ] Participate in Kaggle competitions with Gen AI
- [ ] Contribute to open-source AI agent frameworks
- [ ] Apply for the upcoming 5-Day AI Agents Intensive course

---

## 📄 Course Credits
**Instructors**: Anant Nawalgaria, Antonio Gulli, Mark McDonald, Polong Lin, Paige Bailey  
**Contributors**: Google AI Team and Expert Speakers  
**Course**: Google's 5-Day Gen AI Intensive (March 31 - April 4, 2025)

<div align="center">

**⭐ Star this repo if you found the course journey helpful! ⭐**  
*Building AI solutions one day at a time* 🚀

</div>
