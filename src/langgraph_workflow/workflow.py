"""LangGraph-based agentic RAG pipeline for Abhinav Digital Twin"""

from typing import TypedDict, Literal, Annotated, Optional, List
from datetime import datetime, timedelta
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from src.rag.router import route_query
from src.rag.retrievers import retrieve_documents
from src.rag.generator import generate_response, build_conversation_context
from src.rag.utils import get_llm, validate_api_key
from src.tools.tools import get_calendar_tools
from src.data_validation.pydantic_models import ChatMessage


class RAGState(TypedDict):
    """State for the agentic RAG graph."""
    query: str
    collection: str
    query_type: str
    k: int
    strategy: str
    routing_reasoning: str
    documents: list[Document]
    answer: str
    error: str | None
    mode: Literal["rag", "calendar"]
    conversation_history: Optional[List[ChatMessage]]
    tools_called: Optional[List[str]]


def routing_node(state: RAGState) -> RAGState:
    """Node to route query to appropriate collection for RAG retrieval.
    
    Routes query to personal/professional collection and determines
    query type (fact-based vs comprehensive) and k value.
    """
    # Normal RAG routing
    llm = get_llm(temperature=0.0)
    route_decision = route_query(state["query"], llm=llm)
    
    state["collection"] = route_decision["collection"]
    state["query_type"] = route_decision["type"]
    state["k"] = route_decision["k"]
    state["routing_reasoning"] = route_decision.get("reasoning", "")
    state["mode"] = "rag"
    
    return state


def retrieval_node(state: RAGState) -> RAGState:
    """Node to retrieve documents from vector store."""
    documents = retrieve_documents(
        query=state["query"],
        collection=state["collection"],
        strategy=state["strategy"],
        k=state["k"]
    )
    
    state["documents"] = documents
    
    return state


def generator_node(state: RAGState) -> RAGState:
    """Node to generate response from retrieved documents using RAG generator.
    
    Uses generate_response from src.rag.generator which handles conversation history.
    Conversation history is always passed through the RAG pipeline.
    """
    from src.rag.generator import generate_response
    
    llm = get_llm(model="gpt-4o-mini", temperature=0.1)
    conversation_history = state.get("conversation_history")
    
    answer = generate_response(
        documents=state["documents"],
        query=state["query"],
        collection=state["collection"],
        llm=llm,
        conversation_history=conversation_history
    )
    
    state["answer"] = answer
    state["error"] = None
    
    return state


def unified_agent_node(state: RAGState) -> RAGState:
    """Unified agent node that lets LLM decide via tool calling.
    
    Stage 1 only:
    1. Call LLM with tools (no RAG docs) to check if calendar tools are needed
    2. If calendar tools called → execute and return answer (skip RAG pipeline)
    3. If no calendar tools → set flag to continue to RAG pipeline (routing → retrieval → generator)
    
    Conversation history is always passed and stored for both calendar and RAG queries.
    """
    try:
        # Get calendar tools
        calendar_tools = get_calendar_tools()
        tool_map = {tool.name: tool for tool in calendar_tools}
        
        # Get LLM with calendar tools bound
        llm = get_llm(model="gpt-4.1-nano", temperature=0.1)
        llm_with_tools = llm.bind_tools(calendar_tools)
        
        # Current date and time in PST for calendar tool args
        from zoneinfo import ZoneInfo
        pst = ZoneInfo("America/Los_Angeles")
        now_pst = datetime.now(pst)
        current_date_str = now_pst.strftime("%Y-%m-%d")
        current_time_pst_str = now_pst.strftime("%Y-%m-%d %H:%M %Z")

        # STAGE 1: Call LLM with tools but NO RAG documents
        system_prompt_stage1 = f"""You are Abhinav's digital twin with access to Google Calendar tools and a daily news tool.

Today's date is {current_date_str}. Current time in PST is {current_time_pst_str}. When the user says "today", "tomorrow", "this week", or "this weekend", interpret them relative to this date.

For calendar: If the user gives a time without timezone (e.g. "9am", "2pm tomorrow"), treat it as PST. Pass start_time and end_time in ISO 8601 without Z: use the date and local time, e.g. "2026-02-15T09:00:00" for 9am PST on that date. The tool will interpret times without timezone as PST. Do NOT use Z (UTC) for user-intended local times.

When you create a calendar event, you MUST include the event link (htmlLink or the "View in Google Calendar" URL from the tool result) in your reply so the user can open it.

If the user asks about calendar events, schedule, plans, or availability, you MUST call calendar tools. Do NOT guess.

When the user wants to DELETE or CANCEL an event and gives a time or description (e.g. "delete breakfast tomorrow 6am"): you MUST first call get_calendar_events for that time range to get the event list, then call delete_calendar_event(event_id) with the matching event's "id". Do not say the event was deleted until delete_calendar_event has been called and returned success.

If the user asks for a news update, daily brief, or news summary, you MUST call get_daily_news_summary. Do NOT make up news.

If the query is NOT about calendar/schedule or news, do NOT call any tools; respond normally (documents may be used in the next step)."""
        
        messages_stage1 = [
            SystemMessage(content=system_prompt_stage1)
        ]
        
        # Add conversation history if available
        conversation_history = state.get("conversation_history")
        if conversation_history and len(conversation_history) > 0:
            try:
                conversation_summary = build_conversation_context(conversation_history)
                if conversation_summary:
                    messages_stage1.append(SystemMessage(content=f"Previous conversation context:\n{conversation_summary}"))
            except Exception:
                pass  # Continue without conversation history if it fails
        
        # Add user query
        messages_stage1.append(HumanMessage(content=state["query"]))
        
        # First LLM call - check if it wants calendar tools
        response_stage1 = llm_with_tools.invoke(messages_stage1)
        tool_calls = getattr(response_stage1, 'tool_calls', None) or []
        
        # Check if any tools were called (calendar or news)
        tool_called = any(
            (tc.get('name', '') if isinstance(tc, dict) else getattr(tc, 'name', '')) in tool_map
            for tc in tool_calls
        )
        
        if tool_called:
            # Tools were called - execute them and return answer (skip RAG)
            called_tool_names = []
            for tc in tool_calls:
                name = tc.get('name', '') if isinstance(tc, dict) else getattr(tc, 'name', '')
                if name:
                    called_tool_names.append(name)
            state["tools_called"] = called_tool_names
            print(f"[DEBUG] Tools called: {called_tool_names} - skipping RAG retrieval")
            messages = messages_stage1 + [response_stage1]
            
            # Execute tool calls; keep raw news digest so we can return it with links intact
            news_digest_raw: Optional[str] = None
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id", "")
                else:
                    tool_name = getattr(tool_call, "name", "")
                    tool_args = getattr(tool_call, "args", {})
                    tool_id = getattr(tool_call, "id", "")
                
                if tool_name in tool_map:
                    try:
                        tool_result = tool_map[tool_name].invoke(tool_args)
                        content = str(tool_result)
                        if tool_name == "get_daily_news_summary":
                            news_digest_raw = content  # preserve markdown with links
                        messages.append(
                            ToolMessage(
                                content=content,
                                tool_call_id=tool_id
                            )
                        )
                    except Exception as e:
                        messages.append(
                            ToolMessage(
                                content=f"Error executing {tool_name}: {str(e)}",
                                tool_call_id=tool_id
                            )
                        )
            
            # Get final answer after tool execution
            max_iterations = 5
            final_response = response_stage1
            for iteration in range(max_iterations - 1):  # Already did one iteration
                final_response = llm_with_tools.invoke(messages)
                messages.append(final_response)
                
                tool_calls_final = getattr(final_response, 'tool_calls', None) or []
                if tool_calls_final:
                    # Execute additional tool calls if needed
                    for tool_call in tool_calls_final:
                        if isinstance(tool_call, dict):
                            tool_name = tool_call.get("name", "")
                            tool_args = tool_call.get("args", {})
                            tool_id = tool_call.get("id", "")
                        else:
                            tool_name = getattr(tool_call, "name", "")
                            tool_args = getattr(tool_call, "args", {})
                            tool_id = getattr(tool_call, "id", "")
                        
                        if tool_name in tool_map:
                            try:
                                tool_result = tool_map[tool_name].invoke(tool_args)
                                content = str(tool_result)
                                if tool_name == "get_daily_news_summary":
                                    news_digest_raw = content
                                messages.append(
                                    ToolMessage(
                                        content=content,
                                        tool_call_id=tool_id
                                    )
                                )
                            except Exception as e:
                                messages.append(
                                    ToolMessage(
                                        content=f"Error executing {tool_name}: {str(e)}",
                                        tool_call_id=tool_id
                                    )
                                )
                else:
                    break
            
            # Use raw news digest when available so links are preserved (LLM often drops them)
            if news_digest_raw is not None:
                state["answer"] = news_digest_raw
            else:
                answer = final_response.content if hasattr(final_response, 'content') else str(final_response)
                state["answer"] = answer
            state["documents"] = []  # No RAG documents used
            state["mode"] = "calendar"
            return state
        
        # No calendar tools called - continue to RAG pipeline
        print(f"[DEBUG] No calendar tools called - continuing to RAG pipeline")
        state["mode"] = "rag"
        # State will be passed to routing_node next
        return state
        
    except Exception as e:
        import traceback
        error_msg = f"Error in unified agent: {str(e)}"
        # If calendar tools fail, fall back to RAG pipeline
        print(f"[DEBUG] Error in unified_agent_node: {error_msg}, falling back to RAG pipeline")
        state["mode"] = "rag"
        state["error"] = None
        # Continue to RAG pipeline (routing_node will handle it)
        return state


def create_rag_graph() -> StateGraph:
    """Create the agentic RAG graph with langgraph.
    
    Graph flow:
    START -> unified_agent_node -> (conditional)
      ├─ calendar tools called → END (answer returned, not stored in history)
      └─ no calendar tools → routing_node → retrieval_node → generator_node → END
    
    The unified agent node checks if LLM wants calendar tools.
    If calendar tools are called, execute and return (skip RAG pipeline).
    If no calendar tools, continue to RAG pipeline:
    - routing_node: Routes to personal/professional, determines query type
    - retrieval_node: Retrieves documents based on routing
    - generator_node: Generates response using RAG (with conversation history)
    
    Conversation history is only passed through RAG pipeline, not calendar queries.
    
    Returns:
        Compiled StateGraph
    """
    # Create graph
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("agent", unified_agent_node)
    workflow.add_node("route", routing_node)
    workflow.add_node("retrieve", retrieval_node)
    workflow.add_node("generate", generator_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Conditional routing from unified_agent_node
    def route_after_calendar_check(state: RAGState):
        """Route based on whether calendar tools were called."""
        if state.get("mode") == "calendar":
            return END  # Calendar tools called, return answer
        return "route"  # No calendar tools, continue to RAG pipeline
    
    workflow.add_conditional_edges("agent", route_after_calendar_check)
    
    # RAG pipeline flow
    workflow.add_edge("route", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    # Compile
    app = workflow.compile()
    
    return app


def agentic_rag(
    query: str,
    strategy: Literal["semantic"] = "semantic",
    verbose: bool = True,
    conversation_history: Optional[List[ChatMessage]] = None
) -> str:
    """Execute agentic RAG pipeline using langgraph.
    
    Pipeline steps (graph-based):
    1. Route query to appropriate collection (personal/professional/both)
    2. Retrieve documents using specified strategy
    3. Unified agent node with access to both RAG documents and calendar tools
    4. LLM automatically decides when to use calendar tools vs RAG documents
    
    Args:
        query: User query string
        strategy: Retrieval strategy - currently only "semantic" supported
        verbose: Print routing decisions and progress
        conversation_history: Optional conversation history for context
        
    Returns:
        Generated answer string
    """
    # Validate API key
    validate_api_key()
    
    if verbose:
        print(f"\nProcessing query: '{query}'")
        print(f"Retrieval strategy: {strategy}")
        print(f"Using langgraph agentic workflow...\n")
    
    # Create initial state
    initial_state = {
        "query": query,
        "collection": "",
        "query_type": "",
        "k": 3,
        "strategy": strategy,
        "routing_reasoning": "",
        "documents": [],
        "answer": "",
        "error": None,
        "mode": "rag",
        "conversation_history": conversation_history,
        "tools_called": []
    }
    
    # Create and run graph
    app = create_rag_graph()
    
    # Execute graph
    final_state = app.invoke(initial_state)
    
    if verbose:
        print("="*80)
        print("ROUTING DECISION")
        print("="*80)
        print(f"Collection: {final_state['collection']}")
        print(f"Type: {final_state['query_type']}")
        print(f"Top-K: {final_state['k']}")
        print(f"Retrieved {len(final_state['documents'])} documents")
        print(f"Reasoning: {final_state['routing_reasoning']}")
        print("="*80 + "\n")
    
    return final_state["answer"]


def agentic_rag_with_metadata(
    query: str,
    strategy: Literal["semantic"] = "semantic",
    verbose: bool = False,
    conversation_history: Optional[List[ChatMessage]] = None
) -> dict:
    """Execute agentic RAG and return full metadata.
    
    Args:
        query: User query string
        strategy: Retrieval strategy
        verbose: Print progress
        conversation_history: Optional conversation history for context
        
    Returns:
        Dictionary with answer, routing info, documents, and metadata
    """
    validate_api_key()
    
    initial_state = {
        "query": query,
        "collection": "",
        "query_type": "",
        "k": 3,
        "strategy": strategy,
        "routing_reasoning": "",
        "documents": [],
        "answer": "",
        "error": None,
        "mode": "rag",
        "conversation_history": conversation_history,
        "tools_called": []
    }
    
    app = create_rag_graph()
    final_state = app.invoke(initial_state)
    
    # Extract source information
    sources = []
    for doc in final_state.get('documents', []):
        source_info = {
            'filename': doc.metadata.get('source', 'Unknown'),
            'page': doc.metadata.get('page', 'N/A'),
            'category': doc.metadata.get('category', 'N/A')
        }
        if source_info not in sources:
            sources.append(source_info)
    
    return {
        'answer': final_state['answer'],
        'query': final_state['query'],
        'collection': final_state.get('collection', ''),
        'query_type': final_state.get('query_type', ''),
        'k': final_state.get('k', 0),
        'routing_reasoning': final_state.get('routing_reasoning', ''),
        'num_documents': len(final_state.get('documents', [])),
        'sources': sources,
        'strategy': strategy,
        'error': final_state.get('error'),
        'tools_called': final_state.get('tools_called', [])
    }


if __name__ == "__main__":
    # Test queries
    test_queries = [
        "Where did I study?",
        "What research have I worked on?",
        "Summarize my professional experience"
    ]
    
    print("="*80)
    print("VIVEN DIGITAL TWIN - LANGGRAPH AGENTIC RAG TEST")
    print("="*80)
    
    for query in test_queries:
        print("\n" + "="*80)
        answer = agentic_rag(query, strategy="semantic", verbose=True)
        print("\nFINAL ANSWER:")
        print("-" * 80)
        print(answer)
        print("="*80)
