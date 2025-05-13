import os
import streamlit as st
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# 0) Retrieve OpenAI API key from Streamlit secrets or environment
openai_api_key="#replace with openai key"


# 1) Initialize Neo4j connection and GraphCypherQAChain
graph = Neo4jGraph(
    url="bolt://localhost:11003",
    username="neo4j",
    password="pass"
)
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.6,
    openai_api_key=openai_api_key
)

# Define the Cypher-only prompt
cypher_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
-- **GENERATE ONLY THE CYPHER QUERY BELOW** (no explanation, no comments).
-- User asked: {question}

WITH "{question}" AS q
// split on ' in ' and take the sector part
WITH split(toLower(q), " in ")[1] AS sector
MATCH (i:Industry)
  WHERE toLower(i.name) = sector
     OR toLower(i.name) CONTAINS sector
     OR toLower(i.description) CONTAINS sector
WITH i
MATCH (i)-[:CONNECTED_TO]-(other:Industry)
RETURN other.name AS connectedIndustry
LIMIT 10
"""
)

# Define the follow-up QA prompt
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You’re an expert advisor. The user asked:
  "{question}"

Here are the raw Cypher results (which may be empty):
{context}

Using those results and your own knowledge, give **five** concrete ideas or next‑steps in that sector.
Number your list 1–5.
"""
)

# Build the QA chain
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=cypher_prompt,
    qa_prompt=qa_prompt,
    verbose=True,
    allow_dangerous_requests=True
)

# 2) Inject CSS for fixed bottom input and buttons
st.markdown(
    """
    <style>
      div.block-container { padding-bottom: 180px; }
      .stTextInput > label { display: none; }
      .stTextInput > div { height: 0!important; margin: 0!important; padding: 0!important; overflow: visible; }
      .stTextInput > div > div > input {
        position: fixed!important;
        bottom: 80px!important;
        left: 50%!important;
        transform: translateX(-50%)!important;
        width: 50%!important;
        min-width: 300px!important;
        max-width: 600px!important;
        padding: 12px 16px!important;
        border: 2px solid #555!important;
        border-radius: 8px!important;
        font-size: 16px!important;
        background: #333!important;
        color: #fff!important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.4)!important;
        z-index: 1000!important;
      }
      div.block-container [data-testid="stButton"][data-key="refresh"] > button {
        position: fixed!important;
        bottom: 80px!important;
        left: calc(50% - 25vw - 110px)!important;
        width: 100px!important;
        height: 48px!important;
        border-radius: 8px!important;
        background-color: #555!important;
        color: #fff!important;
        font-size: 14px!important;
        font-weight: 600!important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.4)!important;
        z-index: 1001!important;
      }
      div.stButton > button:last-of-type {
        position: fixed!important;
        bottom: 80px!important;
        left: calc(50% + 310px)!important;
        width: 100px!important;
        height: 48px!important;
        border-radius: 8px!important;
        background-color: #555!important;
        color: #fff!important;
        font-size: 16px!important;
        font-weight: 600!important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.4)!important;
        z-index: 1000!important;
      }
      div.stButton > button:hover { background-color: #444!important; }
      @media (max-width: 768px) {
        .stTextInput > div > div > input { width: 80%!important; bottom: 140px!important; }
        div.stButton > button:first-of-type { bottom: 80px!important; right: auto!important; left: 50%!important; transform: translateX(-110%)!important; }
        div.stButton > button:last-of-type { bottom: 80px!important; left: 50%!important; transform: translateX(10%)!important; }
      }
    </style>
    """, unsafe_allow_html=True
)

# 3) App title and session state
st.title("🗨️ Industry Relationship Knowledge Graph")
if "messages" not in st.session_state:
    st.session_state.messages = []
if "message" not in st.session_state:
    st.session_state.message = ""

# 4) Clear history callback
def clear_history():
    st.session_state.messages = []
    st.session_state.message = ""

# 5) Send message callback with LLM integration
def send_callback():
    user_input = st.session_state.message.strip()
    if not user_input:
        return
    st.session_state.messages.append({"role": "user", "content": user_input})
    try:
        # Invoke the GraphCypherQAChain
        ai_response = chain.invoke({"query": user_input})
    except Exception as e:
        ai_response = f"Error: {e}"
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    st.session_state.message = ""

# 6) UI buttons and chat history
st.button("🔄 Refresh Chat", on_click=clear_history, key="refresh")
for msg in st.session_state.messages:
    align = "right" if msg["role"] == "user" else "left"
    bg = "#007acc" if msg["role"] == "user" else "#333"
    st.markdown(
        f"""
        <div style="text-align:{align}; margin:8px 0;">
          <span style="display:inline-block; background:{bg}; color:#fff; padding:8px 12px; border-radius:12px; max-width:80%; word-wrap:break-word; box-shadow:0 2px 4px rgba(0,0,0,0.3);">{msg['content']}</span>
        </div>
        """, unsafe_allow_html=True
    )

# 7) Input and send button
st.text_input(
    "",
    key="message",
    placeholder="Type your message here…",
    on_change=send_callback
)
st.button("Send", on_click=send_callback, key="send")
