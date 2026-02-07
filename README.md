# 📊 Sales Analytics Chatbot

AI-powered sales data assistant built with LangGraph, Gemini 2.5 Pro, and FastAPI.

## Features

✅ **Natural Language Queries** - Ask questions in plain English
✅ **Conversation Memory** - Understands context and follow-up questions  
✅ **Smart SQL Generation** - Automatically generates queries from your questions
✅ **Beautiful UI** - Modern, responsive chat interface
✅ **Real-time Analysis** - Instant insights from 2,823 sales records

## Data Overview

- **Time Period**: 2003-2004
- **Records**: 2,823 sales transactions
- **Products**: Motorcycles, Classic Cars, Trucks, Vintage Cars, Planes, Ships, Trains
- **Territories**: NA, EMEA, APAC, Japan
- **Deal Sizes**: Small, Medium, Large

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your API key from: https://makersuite.google.com/app/apikey

### 3. Run the Server

```bash
python app.py
```

Open http://localhost:8000 in your browser

## Example Questions

### Revenue Analysis
- "What are total sales?"
- "Show me sales by product line"
- "Which products generate the most revenue?"

### Customer Insights
- "Top 10 customers by revenue"
- "Show me customers in USA"
- "Which customer has the most orders?"

### Product Analysis
- "How many Classic Cars were sold?"
- "Compare Motorcycles vs Planes sales"
- "Show me all product lines"

### Geographic Analysis
- "Sales by country"
- "Top 5 countries by revenue"
- "Show me European sales"

### Deal Analysis
- "Show me large deals"
- "Average deal size by product"
- "How many small vs medium vs large deals?"

### Status & Trends
- "How many shipped orders?"
- "Show disputed orders"
- "Sales by quarter"

## Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Query Analyzer  │ ← Understands intent
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SQL Generator   │ ← Creates SQL query
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Query Executor  │ ← Runs on CSV data
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Response Gen    │ ← Natural language response
└─────────────────┘
```

## Tech Stack

- **LLM**: Google Gemini 2.5 Pro
- **Framework**: LangGraph (conversation flow)
- **Backend**: FastAPI
- **Frontend**: Vanilla JS + Modern CSS
- **Data**: Pandas + SQLite (in-memory)

## Project Structure

```
portconnect/
├── app.py                    # FastAPI server
├── chatbot.py                # LangGraph chatbot logic
├── sales_data_sample.csv     # Sales data
├── templates/
│   └── index.html            # Chat UI
├── requirements.txt          # Dependencies
├── .env                      # API keys
└── README.md                 # This file
```

## Memory Feature

The chatbot remembers conversation context:

```
You: "Show me top customers"
Bot: [Shows top 10 customers]

You: "What about their orders?"
Bot: [Shows orders for those customers - remembers context!]

You: "Focus on Euro Shopping Channel"
Bot: [Shows details for that specific customer]
```

## Customization

### Change Data Source

Edit `chatbot.py`:

```python
cls.sales_df = pd.read_csv("your_data.csv", encoding='latin1')
```

### Modify Prompts

Update prompts in `chatbot.py`:
- `QUERY_ANALYZER_PROMPT`
- `SQL_GENERATOR_PROMPT`
- `RESPONSE_GENERATOR_PROMPT`

### Adjust UI

Edit `templates/index.html` for styling and layout changes.

## Troubleshooting

### API Key Error
```
ValueError: GEMINI_API_KEY not found
```
**Solution**: Create `.env` file with your API key

### CSV Encoding Error
```
UnicodeDecodeError
```
**Solution**: The code uses `encoding='latin1'` - adjust if needed

### Port Already in Use
```
Address already in use
```
**Solution**: Change port in `app.py`: `uvicorn.run(app, port=8001)`

## Performance

- **Response Time**: 2-5 seconds per query
- **Memory Usage**: ~50MB (data loaded once)
- **Concurrent Users**: Supports multiple sessions

## License

MIT License - Feel free to use and modify!

## Credits

Built with ❤️ using:
- Google Gemini 2.5 Pro
- LangGraph
- FastAPI
- Pandas
