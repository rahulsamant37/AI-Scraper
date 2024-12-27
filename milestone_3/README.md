# Workflow Overview

Below is a flowchart that outlines the high-level workflow of our project, from user interface setup through data scraping, text chunking, AI model processing, and result saving. This provides a clear visual representation of each stage in the pipeline.

```mermaid
graph TD
    A[Start] --> B[User Interface Setup]
    B -->|Initialize| C[Streamlit Components]
    
    subgraph UI_Components[User Interface Components]
        C --> D1[URL Input Field]
        C --> D2[Model Selection Dropdown]
        C --> D3[Fields Input Tags]
        C --> D4[Chunk Size Slider]
        C --> D5[Chunk Overlap Slider]
    end
    
    UI_Components --> E[Scrape Button Clicked]
    
    E --> F[Setup Selenium]
    F -->|Configure| F1[Set User Agent]
    F -->|Configure| F2[Set Headless Options]
    F -->|Initialize| F3[Chrome WebDriver]
    
    F3 --> G[Fetch HTML]
    G -->|Selenium Actions| G1[Load Page]
    G1 --> G2[Scroll Page]
    G2 --> G3[Get Page Source]
    
    G3 --> H[Clean HTML]
    H -->|BeautifulSoup| H1[Remove Headers]
    H1 -->|BeautifulSoup| H2[Remove Footers]
    
    H2 --> I[Convert to Markdown]
    I -->|html2text| I1[Raw Markdown Text]
    
    I1 --> J[Text Chunking]
    J -->|RecursiveCharacterTextSplitter| J1[Text Chunks]
    J1 -->|Size: User Defined| J2[Overlapping Chunks]
    
    subgraph Dynamic_Models[Create Dynamic Models]
        K[Create Models]
        K -->|From User Fields| K1[Dynamic Listing Model]
        K -->|Container| K2[Listings Container Model]
        K1 --> K2
    end
    
    J2 --> L[Process Chunks]
    Dynamic_Models --> L
    
    L --> M{Select AI Model Processing}
    
    subgraph AI_Processing[AI Model Processing]
        M -->|OpenAI| N1[GPT Processing]
        M -->|Google| N2[Gemini Processing]
        M -->|Local| N3[Llama Processing]
        M -->|Groq| N4[Groq Processing]
        
        N1 & N2 & N3 & N4 -->|Generate| O1[JSON Output]
        O1 -->|Parse| O2[Structured Data]
    end
    
    O2 --> P[Merge Results]
    P -->|Remove Duplicates| P1[Combined Data]
    
    P1 --> Q[Calculate Metrics]
    Q -->|Count| Q1[Input Tokens]
    Q -->|Count| Q2[Output Tokens]
    Q -->|Calculate| Q3[Total Cost]
    
    subgraph Save_Results[Save and Display Results]
        R[Format Results]
        R --> R1[JSON File]
        R --> R2[CSV/Excel File]
        R --> R3[Markdown File]
        R --> R4[Display in UI]
        R --> R5[Show Token Usage]
    end
    
    Q3 --> R
    
    style A fill:#f9f,stroke:#333
    style E fill:#bbf,stroke:#333
    style R4 fill:#bfb,stroke:#333
    style Dynamic_Models fill:#ffd,stroke:#333
    style AI_Processing fill:#dff,stroke:#333
    style Save_Results fill:#dfd,stroke:#333
