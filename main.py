# Load environment variables from a .env file.
from config import settings

# Define structured output models using Pydantic and typing
from pydantic import BaseModel
from typing import Literal, TypedDict

# Langchain imports that we will use to interact with Groq models
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# Custom tools that we will use. These are pulled from our tools.py
from tools import search_and_scrape as scrape_tool, search_website as search_tool, save_to_txt as save_tool

# Pulling our Gemini API key from our .env file.
GROQ_API_KEY = settings.GROQ_API_KEY

# Define the structure of each lead in the output
class LeadResponse(BaseModel):
    company: str
    contact_info: str
    email: str
    summary: str
    outreach_message: str
    tools_used: list[str]

# Define a list structure to hold multiple leads
class LeadResponseList(BaseModel):
    leads: list[LeadResponse]

# Determining which AI model we will use, in this case, Gemini-2.5-flash
llm = ChatGroq(model="openai/gpt-oss-20b", 
            temperature=0.7,
            api_key=GROQ_API_KEY)

# Tell Gemini how to format the response using the Pydantic schema
parser = llm.with_structured_output(LeadResponseList, method="json_mode")

#Agent status (memory)
class AgentState(TypedDict):
    query: str
    raw_companies: list
    enriched_leads: LeadResponseList
    saved_status: str

# The main part here. This are the nodes we define that the agent will follow
def scrape_companies_node(state: AgentState):
    search = search_tool.invoke({"query": state["query"]})
    print("CONTENIDO REAL DE LA BÚSQUEDA:", search[:500])
    companies = scrape_tool.invoke({"query": str(search)})
    return {"raw_companies": companies}

def format_node(state:AgentState):
    prompt = ChatPromptTemplate.from_messages(
        [
            ( 
                "system",
                """
                You are a strict data formatting assistant.
                Your ONLY task is to analyze the raw business data provided by the user and populate the requested JSON/Pydantic schema.
                
                CRITICAL RULES:
                1. Do NOT reply with conversational text, greetings, or explanations.
                2. You MUST output ONLY the structured LeadResponseList schema based on the data.
                3. Ensure all JSON keys EXACTLY match the Pydantic properties using snake_case.
                4. CRITICAL: The property 'tools_used' MUST be a valid JSON array of strings (e.g., ["tool1", "tool2"]), NOT a single string separated by commas.
                5. If no clear companies are found in the input, return the structure with an empty list of leads: {{"leads": []}}.
                
                You must EXACTLY extract and map the properties using these precise JSON keys:
                    - company
                    - contact_info
                    - summary
                    - email
                    - outreach_message
                    - tools_used
                """,
            ),
            ("human", "Analyze the following raw data and fill the schema. If empty, return an empty leads list. Data: {companies_data}")
        ]
    )
    full_prompt = prompt.invoke({"companies_data": state["raw_companies"]})
    leads = parser.invoke(full_prompt)

    #flow control
    if len(leads.leads) == 0:
        return Command(
        update={"saved_status": "no data"},
        goto=END
        )
    return Command(
        update={"enriched_leads": leads},
        goto="saver"
    )

def save_result_node(state:AgentState):
    save_tool.invoke({"data": state["enriched_leads"].model_dump_json()})
    return {"saved_status": "data fetched"}

#Graph construction
workflow = StateGraph(AgentState)

workflow.add_node("scrapper", scrape_companies_node)
workflow.add_node("formatter", format_node)
workflow.add_node("saver", save_result_node)

#workflow
workflow.add_edge(START, "scrapper")
workflow.add_edge("scrapper", "formatter")
workflow.add_edge("saver", END)

app = workflow.compile()


if __name__ == "__main__":
    # Define the query that kicks off the lead generation process
    query = "IT Support and Managed Services companies in Vancouver British Columbia Canada"

    try:
        # Run the agent with the query
        response = app.invoke({"query": query})
        leads_found = response.get("enriched_leads")
        
        if leads_found:
            print("\nLeads processed and saved!")
            print(leads_found)
        else:
            print(f"\nNo leads found")
            print(f"Agent status: {response.get('saved_status', 'without status')}")
    except Exception as e:
        print("Error parsing response", e)