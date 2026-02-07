import os
import json
import sqlite3
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()

# -----------------------------
# Logging Configuration
# -----------------------------
class Logger:
    """Centralized logging for the chatbot with consistent formatting."""
    
    DEBUG_MODE = True
    
    ICONS = {
        "query": "🔍",
        "sql": "🔧",
        "execute": "⚡",
        "response": "💬",
        "success": "✅",
        "error": "❌",
        "warning": "⚠️",
        "info": "ℹ️",
        "debug": "🐛",
    }
    
    @classmethod
    def _get_timestamp(cls) -> str:
        return datetime.now(ZoneInfo("Pacific/Auckland")).strftime("%H:%M:%S")
    
    @classmethod
    def header(cls, node_name: str, icon_key: str = "info"):
        icon = cls.ICONS.get(icon_key, "📌")
        print(f"\n{icon} [{cls._get_timestamp()}] [{node_name}]")
        print("-" * 50)
    
    @classmethod
    def info(cls, message: str):
        print(f"   {cls.ICONS['info']} {message}")
    
    @classmethod
    def success(cls, message: str):
        print(f"   {cls.ICONS['success']} {message}")
    
    @classmethod
    def error(cls, message: str):
        print(f"   {cls.ICONS['error']} {message}")
    
    @classmethod
    def warning(cls, message: str):
        print(f"   {cls.ICONS['warning']} {message}")
    
    @classmethod
    def debug(cls, label: str, value: any):
        if cls.DEBUG_MODE:
            str_value = str(value)
            if len(str_value) > 200:
                str_value = str_value[:200] + "..."
            print(f"   {cls.ICONS['debug']} {label}: {str_value}")
    
    @classmethod
    def key_value(cls, label: str, value: any):
        print(f"   → {label}: {value}")
    
    @classmethod
    def sql(cls, query: str):
        clean_query = query.strip()
        if len(clean_query) > 300:
            print(f"   📝 SQL Query:\n      {clean_query[:300]}...")
        else:
            print(f"   📝 SQL Query:\n      {clean_query}")

# Get API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

print(f"🔑 API Key loaded: {GEMINI_API_KEY[:20]}...")

# Initialize the model
try:
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=GEMINI_API_KEY,
        temperature=0.1
    )
    print("✅ Gemini model initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize Gemini model: {e}")
    raise

# -----------------------------
# Global Data Store
# -----------------------------
class DataStore:
    """Singleton to store sales data"""
    sales_df: pd.DataFrame = None
    loaded_at: datetime = None
    
    @classmethod
    def load_data(cls):
        """Load sales data from CSV"""
        if cls.sales_df is None:
            cls.sales_df = pd.read_csv("sales_data_sample.csv", encoding='latin1')
            cls.loaded_at = datetime.now(ZoneInfo("Pacific/Auckland"))
            Logger.success(f"Loaded {len(cls.sales_df)} sales records")
        return cls.sales_df


# Load data on startup
DataStore.load_data()


# -----------------------------
# State Definition
# -----------------------------
class ChatState(TypedDict):
    """State for the sales chatbot with conversation memory."""
    messages: Annotated[list, add_messages]
    sql_query: str
    query_result: str
    user_query: str


# -----------------------------
# Prompt Templates
# -----------------------------

QUERY_ANALYZER_PROMPT = """You are a sales data analyst assistant. You have access to conversation history to understand context.

The user is asking about sales data. Analyze their question and understand what they want to know.

Available data includes:
- Orders and sales amounts
- Products (Motorcycles, Classic Cars, Trucks and Buses, Vintage Cars, Planes, Ships, Trains)
- Customers and their details
- Countries and territories (USA, France, UK, etc.)
- Deal sizes (Small, Medium, Large)
- Order status (Shipped, Disputed, In Process, Cancelled, On Hold, Resolved)
- Time periods (2003-2004 data)
- Quantities, prices, and revenue

Just acknowledge you understand the query. Output: "Understood: [brief summary of what user wants]"

User question: """


SQL_GENERATOR_PROMPT = """You are an expert SQL query generator for a sales database.

IMPORTANT: You have access to conversation history. Use it to understand context and follow-up questions.

The table name is: sales_data

Columns available:
- ORDERNUMBER: Order ID
- QUANTITYORDERED: Quantity of items ordered
- PRICEEACH: Price per item
- ORDERLINENUMBER: Line number in order
- SALES: Total sales amount for this line
- ORDERDATE: Date of order (format: M/D/YYYY H:MM)
- STATUS: Order status (Shipped, Disputed, In Process, Cancelled, On Hold, Resolved)
- QTR_ID: Quarter (1-4)
- MONTH_ID: Month (1-12)
- YEAR_ID: Year (2003, 2004)
- PRODUCTLINE: Product category (Motorcycles, Classic Cars, Trucks and Buses, Vintage Cars, Planes, Ships, Trains)
- MSRP: Manufacturer suggested retail price
- PRODUCTCODE: Product code
- CUSTOMERNAME: Customer name
- PHONE: Customer phone
- ADDRESSLINE1, ADDRESSLINE2: Address
- CITY: City
- STATE: State/Province
- POSTALCODE: Postal code
- COUNTRY: Country name
- TERRITORY: Sales territory (NA, EMEA, APAC, Japan)
- CONTACTLASTNAME, CONTACTFIRSTNAME: Contact person
- DEALSIZE: Deal size category (Small, Medium, Large)

### Context Awareness:
- If user says "show me more" after seeing results, add more columns or remove LIMIT
- If user asks about "that product" or "those customers", reference previous query results
- If user asks follow-up questions, understand they're building on previous context

### Common Queries:
- Total sales: SELECT SUM(SALES) FROM sales_data
- Top products: SELECT PRODUCTLINE, SUM(SALES) as total FROM sales_data GROUP BY PRODUCTLINE ORDER BY total DESC
- By country: SELECT COUNTRY, SUM(SALES) as total FROM sales_data GROUP BY COUNTRY ORDER BY total DESC
- By customer: SELECT CUSTOMERNAME, SUM(SALES) as total FROM sales_data GROUP BY CUSTOMERNAME ORDER BY total DESC LIMIT 10
- By status: SELECT STATUS, COUNT(*) as count FROM sales_data GROUP BY STATUS
- By deal size: SELECT DEALSIZE, SUM(SALES) as total FROM sales_data GROUP BY DEALSIZE
- Time analysis: SELECT YEAR_ID, MONTH_ID, SUM(SALES) as total FROM sales_data GROUP BY YEAR_ID, MONTH_ID ORDER BY YEAR_ID, MONTH_ID

### Output Rules:
- Output ONLY the SQL query
- No explanations, no markdown, no backticks
- Use proper SQL syntax for SQLite
- For dates, use: WHERE ORDERDATE LIKE '%2003%' or WHERE YEAR_ID = 2003

Generate the SQL query:

User question: """


RESPONSE_GENERATOR_PROMPT = """You are a sales analytics assistant. Be helpful, conversational, and insightful.

IMPORTANT: You have access to conversation history for context-aware responses.

User question: {user_query}

Query results:
{results}

RESPONSE STRUCTURE:

1. **SUMMARY** - One clear line answering the question
2. **KEY INSIGHTS** - Show top 5-10 results with formatting
3. **ANALYSIS** - Brief insight or pattern (1-2 sentences)
4. **FOLLOW-UP** - Contextual question based on what they asked

### Formatting Rules:
- Use **bold** for important numbers and names
- Format currency: $1,234.56
- Use bullet points for lists
- Keep it concise and scannable
- Add relevant emojis for visual appeal (📊 💰 🏆 📈)

### Follow-up Logic:
- If showing products: Ask about specific product, time period, or country
- If showing customers: Ask about their orders, products, or trends
- If showing totals: Ask about breakdown by category
- If showing trends: Ask about specific periods or comparisons
- Make it specific to their data, not generic

EXAMPLES:

Query: "Total sales"
Response:
**Total Sales: $10,032,628.85** 💰

That's over 10 million in revenue across all orders!

📊 Want to see this broken down by product line or country?

---

Query: "Top 5 customers"
Response:
**Top 5 Customers by Revenue** 🏆

**Euro Shopping Channel** - $912,294.11
**Mini Gifts Distributors Ltd.** - $654,858.06  
**Australian Collectors, Co.** - $200,995.41
**Muscle Machine Inc** - $197,736.94
**La Rochelle Gifts** - $180,124.90

These 5 customers represent 21% of total revenue!

💡 Want to see what products Euro Shopping Channel buys most?

---

Query: "Sales by product line"
Response:
**Sales by Product Line** 📊

🏍️ **Classic Cars** - $3,919,615.66 (39%)
🏍️ **Vintage Cars** - $1,903,150.84 (19%)
🚛 **Motorcycles** - $1,166,388.34 (12%)
🚛 **Trucks and Buses** - $1,127,789.84 (11%)
✈️ **Planes** - $975,003.57 (10%)
🚢 **Ships** - $714,437.13 (7%)
🚂 **Trains** - $226,243.47 (2%)

Classic Cars dominate with nearly 40% of all sales!

📈 Curious about which countries buy the most Classic Cars?
"""


# -----------------------------
# Helper Functions
# -----------------------------

def extract_json_from_response(response_text: str) -> dict:
    """Extract JSON from LLM response."""
    text = response_text.strip()
    
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    
    start_idx = text.find("{")
    end_idx = text.rfind("}") + 1
    
    if start_idx != -1 and end_idx > start_idx:
        json_str = text[start_idx:end_idx]
        return json.loads(json_str)
    
    return json.loads(text)


# -----------------------------
# Graph Nodes
# -----------------------------

def query_analyzer_node(state: ChatState) -> dict:
    """Analyze user query"""
    Logger.header("Query Analyzer", "query")
    
    # Get the last human message
    last_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_message = msg.content
            break
    
    if not last_message:
        Logger.warning("No user message found")
        return {"user_query": ""}
    
    Logger.key_value("User Query", last_message)
    
    try:
        # Build conversation history
        conversation_messages = [SystemMessage(content=QUERY_ANALYZER_PROMPT)]
        recent_messages = state["messages"][-6:] if len(state["messages"]) > 6 else state["messages"]
        conversation_messages.extend(recent_messages)
        
        Logger.info("Calling Gemini API...")
        response = model.invoke(conversation_messages)
        Logger.success(f"Analysis: {response.content[:100]}")
        
        return {"user_query": last_message}
    except Exception as e:
        Logger.error(f"Query analyzer failed: {str(e)}")
        return {"user_query": last_message}


def sql_generator_node(state: ChatState) -> dict:
    """Generate SQL query"""
    Logger.header("SQL Generator", "sql")
    
    user_query = state.get("user_query", "")
    
    try:
        # Build conversation history
        conversation_messages = [SystemMessage(content=SQL_GENERATOR_PROMPT)]
        recent_messages = state["messages"][-6:] if len(state["messages"]) > 6 else state["messages"]
        conversation_messages.extend(recent_messages)
        
        Logger.info("Generating SQL query...")
        response = model.invoke(conversation_messages)
        
        sql_query = response.content.strip()
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        Logger.sql(sql_query)
        Logger.success("SQL query generated")
        return {"sql_query": sql_query}
    except Exception as e:
        Logger.error(f"SQL generation failed: {str(e)}")
        # Return a simple fallback query
        return {"sql_query": "SELECT COUNT(*) as total_records FROM sales_data"}


def query_executor_node(state: ChatState) -> dict:
    """Execute SQL query"""
    Logger.header("Query Executor", "execute")
    
    sql_query = state.get("sql_query", "")
    
    if DataStore.sales_df is None or DataStore.sales_df.empty:
        Logger.error("No data available")
        return {"query_result": "No data available."}
    
    Logger.key_value("DataFrame Size", f"{len(DataStore.sales_df)} rows")
    
    try:
        conn = sqlite3.connect(":memory:")
        DataStore.sales_df.to_sql("sales_data", conn, index=False, if_exists="replace")
        
        result_df = pd.read_sql_query(sql_query, conn)
        conn.close()
        
        if result_df.empty:
            Logger.warning("No results found")
            result_str = "No results found for your query."
        else:
            result_str = result_df.to_string(index=False, max_rows=20)
            if len(result_df) > 20:
                result_str += f"\n\n... and {len(result_df) - 20} more rows"
            Logger.success(f"Found {len(result_df)} results")
        
        return {"query_result": result_str}
        
    except Exception as e:
        Logger.error(f"SQL execution failed: {str(e)}")
        return {"query_result": f"Error executing query: {str(e)}"}


def response_generator_node(state: ChatState) -> dict:
    """Generate natural language response"""
    Logger.header("Response Generator", "response")
    
    user_query = state.get("user_query", "")
    query_result = state.get("query_result", "")
    
    Logger.info("Generating response...")
    
    try:
        prompt = RESPONSE_GENERATOR_PROMPT.format(
            user_query=user_query,
            results=query_result
        )
        
        # Build conversation history
        conversation_messages = [SystemMessage(content=prompt)]
        recent_messages = state["messages"][-8:] if len(state["messages"]) > 8 else state["messages"]
        conversation_messages.extend(recent_messages)
        
        response = model.invoke(conversation_messages)
        
        Logger.success("Response generated")
        return {"messages": [AIMessage(content=response.content)]}
    except Exception as e:
        Logger.error(f"Response generation failed: {str(e)}")
        # Return a simple fallback response
        return {"messages": [AIMessage(content=f"I found this data:\n\n{query_result}\n\nLet me know if you need more details!")]}


# -----------------------------
# Build the Graph
# -----------------------------

def create_chatbot():
    """Create and compile the chatbot graph"""
    
    graph = StateGraph(ChatState)
    
    # Add nodes
    graph.add_node("query_analyzer", query_analyzer_node)
    graph.add_node("sql_generator", sql_generator_node)
    graph.add_node("query_executor", query_executor_node)
    graph.add_node("response_generator", response_generator_node)
    
    # Set entry point
    graph.set_entry_point("query_analyzer")
    
    # Connect edges
    graph.add_edge("query_analyzer", "sql_generator")
    graph.add_edge("sql_generator", "query_executor")
    graph.add_edge("query_executor", "response_generator")
    graph.add_edge("response_generator", END)
    
    # Create memory
    memory = MemorySaver()
    
    # Compile
    agent = graph.compile(checkpointer=memory)
    
    return agent


# -----------------------------
# Main Chat Loop
# -----------------------------

def chat():
    """Main chat function"""
    print("\n" + "=" * 60)
    print("📊  SALES ANALYTICS CHATBOT")
    print("=" * 60)
    print(f"📅 Today: {datetime.now(ZoneInfo('Pacific/Auckland')).strftime('%A, %d %B %Y')}")
    print(f"📈 Data: 2,823 sales records (2003-2004)")
    print("-" * 60)
    print("💡 Ask about sales, products, customers, countries & more!")
    print("💡 Type 'quit' or 'exit' to end.")
    print("=" * 60)
    
    agent = create_chatbot()
    
    config = {
        "configurable": {
            "thread_id": "user_session_1"
        }
    }
    
    while True:
        try:
            print("\n" + "=" * 60)
            user_input = input("👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "bye", "q"]:
                print("\n" + "=" * 60)
                print("👋 Goodbye!")
                print("=" * 60)
                break
            
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )
            
            print("\n" + "=" * 60)
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    print(f"🤖 Assistant:\n{msg.content}")
                    break
                    
        except KeyboardInterrupt:
            print("\n\n" + "=" * 60)
            print("👋 Goodbye!")
            print("=" * 60)
            break
        except Exception as e:
            Logger.error(f"Error: {str(e)}")
            print("Please try again.")


if __name__ == "__main__":
    chat()
