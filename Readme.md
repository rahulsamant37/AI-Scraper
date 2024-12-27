# 🌟 Universal Web Scraping - AI Processing Pipeline

## 🎓 Infosys Springboard Internship

Excited to present the completion of my Infosys Springboard Internship Milestone 3! This project combines advanced web scraping with AI-powered data processing to create a flexible, robust data extraction pipeline.

## 🚀 Features

- **Multi-Provider AI Integration**: Support for OpenAI, Google Gemini, Llama, and Groq
- **Smart Web Scraping**: Selenium-based scraping with intelligent scroll handling
- **Advanced Text Processing**: Customizable chunking with overlap control
- **Dynamic Model Generation**: Creates data models based on user-defined fields
- **Multiple Export Formats**: JSON, CSV, Excel, and Markdown output options
- **Cost Tracking**: Automated token counting and cost calculation
- **User-Friendly Interface**: Streamlit-based UI with intuitive controls

## 🛠️ Technologies and Tools Used

- **Python**: Core scripting language for logic and data handling
- **Selenium & Playwright**: Dynamic web scraping and content handling
- **Pydantic**: Data processing, model generation, and validation
- **Streamlit**: Creating an intuitive and interactive user interface
- **LangChain & LangSmith**: For structured AI-driven data extraction and workflow tracking
- **ChatGoogleGenerativeAI & ChatGroq**: Enhancing AI model efficiency and accuracy

## 📊 System Architecture

### Dynamic Container Model
```mermaid
graph TD
    A[User Input Fields] -->|Example Input| B["Fields = ['price', 'title', 'description']"]
    
    subgraph Dynamic_Listing_Model[Dynamic Listing Model Creation]
        B --> C[Create Single Item Structure]
        C -->|Creates| D[Pydantic Model]
        D --> E["Single Item Schema:
        {
            'price': string,
            'title': string,
            'description': string
        }"]
    end
    
    subgraph Container_Model[Container Model Creation]
        E --> F[Create Container Structure]
        F -->|Wraps Items| G["Final Schema:
        {
            'listings': [
                {item1},
                {item2},
                {item3},
                ...
            ]
        }"]
    end
    
    H[Real World Example] --> I["User wants to scrape:
    - Product Name
    - Price
    - Rating"]
    
    I --> J["Creates Model:
    {
        'listings': [
            {
                'Product Name': 'iPhone 13',
                'Price': '$799',
                'Rating': '4.5'
            },
            {
                'Product Name': 'Galaxy S21',
                'Price': '$699',
                'Rating': '4.3'
            }
        ]
    }"]
    
    style Dynamic_Listing_Model fill:#ffd,stroke:#333
    style Container_Model fill:#dff,stroke:#333
```

### AI Processing Pipeline
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
    
    style A fill:#f9f,stroke:#333
    style E fill:#bbf,stroke:#333
    style J1 fill:#bfb,stroke:#333
```

## UI
![UI-View](https://github.com/rahulsamant37/AI-Scraper/blob/main/data/UI.png)

## 🔄 Web Scraping Workflow

### 1️⃣ URL Retrieval
- Utilized Selenium with randomized user agents for anonymity
- Automated cookie consent handling for seamless navigation
- Implemented dynamic scrolling to load complex page content
- Captured the full HTML source for further processing

### 2️⃣ HTML Processing
- Cleaned HTML by removing headers, footers, and unnecessary elements
- Converted HTML to markdown format using html2text
- Removed URLs and preserved only meaningful content

### 3️⃣ Data Extraction Strategy
- Generated dynamic models based on user-specified fields using Pydantic
- Integrated multiple AI models for intelligent extraction:
  - GPT-4
  - Gemini-1.5 Flash
  - Llama3.1 (Local/Groq)
- Designed chunk-based processing for large content
- Produced structured JSON outputs

### 4️⃣ Token & Cost Management
- Tracked input and output tokens across models
- Calculated per-model costs with different pricing schemes
- Provided transparent cost metrics

### 5️⃣ Output Options
- Exported results in JSON, CSV, and Excel formats
- Preserved markdown versions for documentation
- Enabled comprehensive logging

## Output
![Ouput-View](https://github.com/rahulsamant37/AI-Scraper/blob/main/data/Output.gif)

## ⚙️ Unique Aspects

- **Adaptive Extraction**: Models adjust dynamically to user specifications
- **Multi-Model Support**: Flexible AI model selection
- **Transparent Token Tracking**: Detailed usage and cost insights

## 🚀 Future Enhancements

- Transitioning to a scalable backend using FASTAPI
- Leveraging LangGraph for graph-based AI visualizations

## 📚 Learning Resources

- Web Scraping: @John Watson Rooney YouTube Channel
- LangChain & AI: **Krish Naik** Sir's Udemy Course
- Documentation: The ultimate teacher!

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/webscraping-ai-pipeline.git
cd webscraping-ai-pipeline

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the application
streamlit run app.py
```

## Resources Followed

Mr. Krish Naik for his comprehensive AI courses
John Watson Rooney for web scraping tutorials
Fellow interns for their collaboration and support

## 📜 License
This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

## 🙏 Acknowledgments
I want to express my sincere gratitude to:
### Infosys Springboard Team

- The mentors who provided invaluable guidance throughout the internship
- The technical team for their support in overcoming challenges
- The program coordinators for organizing this learning opportunity

### Technical Community

- The open-source community for providing excellent tools and libraries
- Stack Overflow contributors for their helpful solutions
- GitHub community for code examples and inspiration


## 🤝 Connect With Me

I'd love to hear your thoughts and suggestions! Feel free to connect and share your ideas.

## Contact Information
For questions or collaboration opportunities:

[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:rahulsamantcoc2@gmail.com)  [![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/rahulsamant37/)  [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rahul-samant-kb37/)
