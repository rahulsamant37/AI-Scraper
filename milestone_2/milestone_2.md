# Web Scraping Project - Milestone 2 📊

A comprehensive web scraping solution developed during my internship that includes implementations for scraping multiple e-commerce and creative platforms.

## 🛠️ Technologies Used

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Selenium](https://img.shields.io/badge/Selenium-43B02A?style=for-the-badge&logo=selenium&logoColor=white)
![Playwright](https://img.shields.io/badge/Playwright-45ba4b?style=for-the-badge&logo=playwright&logoColor=white)
![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup4-43B02A?style=for-the-badge&logo=beautifulsoup4&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![asyncio](https://img.shields.io/badge/asyncio-0D96F6?style=for-the-badge&logo=python&logoColor=white)

## 🎯 Project Overview

This milestone consists of three main tasks:
- Task 4: DealsHeaven Store Scraper
- Task 5: Behance Data Scraper
- Task 6: Combined Web Scraper Interface

## 🌟 Key Features

### Common Features
- User-friendly Streamlit interface
- Multiple export options (CSV, Excel, JSON)
- Search functionality
- Progress tracking
- Error handling and retry mechanisms

### Platform-Specific Features

#### DealsHeaven Scraper
- Multiple scraping implementations (BeautifulSoup, Selenium, Playwright)
- Store selection
- Multi-page scraping
- Concurrent scraping for improved performance

#### Behance Scraper
- Asset and job listing scraping
- Customizable number of items
- Asynchronous operations
- Automatic retries

## 🛠️ Project Structure
```
milestone_2/
├── task_4/            # DealsHeaven Scraper
├── task_5/            # Behance Scraper
├── task_6/            # Combined Interface
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## 🎥 Demo Video

[![Project Demo](https://img.shields.io/badge/Watch_Demo-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](your-video-link-here)

Watch our comprehensive demo showcasing the complete web scraping solution in action! The video demonstrates:

## 📦 Installation

1. Set up the virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Playwright browsers:
```bash
playwright install
```

## 💡 Key Learnings

1. **Web Scraping Techniques**
   - Comparison between different scraping libraries
   - Handling dynamic content loading
   - Managing browser automation

2. **Error Handling**
   - Retry mechanisms for failed requests
   - Graceful handling of missing data
   - Network error management

3. **Performance Optimization**
   - Asynchronous programming
   - Concurrent execution
   - Resource management

4. **UI Development**
   - Streamlit framework usage
   - User experience design
   - Progress tracking implementation

## 🚀 Planned Improvements

1. **URL Filtering Enhancement**
   - Advanced URL parameter handling
   - Filter section in UI
   - Complex query parameters
   - Filter presets

2. **Performance Optimization**
   - Request queuing
   - Rate limiting controls
   - Memory usage optimization
   - Error recovery

3. **User Interface**
   - Dark/light mode toggle
   - Responsive design
   - User preferences
   - Enhanced progress indicators

## ⚙️ Usage

1. Navigate to the desired task directory:
```bash
cd milestone_2/task_X
```

2. Run the Streamlit application:
```bash
streamlit run main.py
```

## ⚠️ Important Notes

- Implement appropriate delays between requests
- Handle session management properly
- Follow website terms of service
- Monitor resource usage

## 🔍 Testing

- Unit tests for core functionality
- Integration tests for UI components
- Error scenario testing
- Performance benchmarking

## 📚 Dependencies

### Core Libraries
![requests](https://img.shields.io/badge/requests-2.28.2-blue?style=flat-square&logo=python&logoColor=white)
![beautifulsoup4](https://img.shields.io/badge/beautifulsoup4-4.11.2-orange?style=flat-square&logo=python&logoColor=white)
![selenium](https://img.shields.io/badge/selenium-4.8.2-green?style=flat-square&logo=selenium&logoColor=white)
![playwright](https://img.shields.io/badge/playwright-1.31.1-red?style=flat-square&logo=playwright&logoColor=white)

### Data Processing
![pandas](https://img.shields.io/badge/pandas-1.5.3-blue?style=flat-square&logo=pandas&logoColor=white)
![numpy](https://img.shields.io/badge/numpy-1.24.2-yellow?style=flat-square&logo=numpy&logoColor=white)

### UI Framework
![streamlit](https://img.shields.io/badge/streamlit-1.19.0-red?style=flat-square&logo=streamlit&logoColor=white)

### Utilities
![asyncio](https://img.shields.io/badge/asyncio-3.4.3-blue?style=flat-square&logo=python&logoColor=white)
![xlsxwriter](https://img.shields.io/badge/xlsxwriter-3.0.9-green?style=flat-square&logo=microsoft-excel&logoColor=white)
![openpyxl](https://img.shields.io/badge/openpyxl-3.1.2-orange?style=flat-square&logo=microsoft-excel&logoColor=white)