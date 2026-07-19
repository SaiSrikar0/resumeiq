import pytest
from typing import List, Optional, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from src.agent.agent import create_agent

class MockReActChatModel(BaseChatModel):
    """
    A custom mock ChatModel to simulate the ReAct tool calling loop.
    It returns a tool call on the first turn based on keywords, 
    and returns a final text answer on the second turn once the tool output is received.
    """
    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        last_message = messages[-1].content
        has_tool_response = any(isinstance(m, ToolMessage) for m in messages)
        
        if not has_tool_response:
            # First turn: map prompt to tool call
            if "score" in last_message.lower() or "breakdown" in last_message.lower():
                tool_call = {
                    "name": "get_score_breakdown",
                    "args": {"resume_id": "REAL_0001", "jd_id": "JD_0001"},
                    "id": "call_1"
                }
                ai_msg = AIMessage(content="", tool_calls=[tool_call])
            elif "missing" in last_message.lower() or "skill" in last_message.lower():
                tool_call = {
                    "name": "get_missing_skills",
                    "args": {"resume_id": "REAL_0001", "jd_id": "JD_0001"},
                    "id": "call_2"
                }
                ai_msg = AIMessage(content="", tool_calls=[tool_call])
            elif "similar" in last_message.lower() or "peer" in last_message.lower():
                tool_call = {
                    "name": "get_similar_resumes",
                    "args": {"resume_id": "REAL_0001"},
                    "id": "call_3"
                }
                ai_msg = AIMessage(content="", tool_calls=[tool_call])
            elif "rewrite" in last_message.lower() or "improve" in last_message.lower():
                tool_call = {
                    "name": "rewrite_suggestion",
                    "args": {"section_text": last_message},
                    "id": "call_4"
                }
                ai_msg = AIMessage(content="", tool_calls=[tool_call])
            else:
                ai_msg = AIMessage(content="I can help you review your score, identify missing skills, find similar resumes, or rewrite bullet points.")
        else:
            # Second turn: tool has run and returned output, now finalize
            tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
            tool_output = tool_msgs[-1].content
            ai_msg = AIMessage(content=f"According to the database: {tool_output}")
            
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    def _llm_type(self) -> str:
        return "mock-react-chat-model"

@pytest.fixture
def agent():
    mock_llm = MockReActChatModel()
    return create_agent(mock_llm)

def test_agent_score_breakdown(agent):
    config = {"configurable": {"thread_id": "test_session_1"}}
    query = "Can you show me my fit score breakdown?"
    
    # We initialize the state with session IDs
    state = {
        "messages": [HumanMessage(content=query)],
        "active_resume_id": "REAL_0001",
        "active_jd_id": "JD_0001"
    }
    
    response = agent.invoke(state, config=config)
    
    # Check that tool call happened
    messages = response["messages"]
    tool_calls = [m.tool_calls for m in messages if hasattr(m, 'tool_calls') and m.tool_calls]
    assert len(tool_calls) == 1
    assert tool_calls[0][0]["name"] == "get_score_breakdown"
    
    # Check final message content is grounded in the tool output
    final_message = messages[-1].content
    assert "According to the database:" in final_message
    assert "Score Breakdown for Resume REAL_0001" in final_message

def test_agent_missing_skills(agent):
    config = {"configurable": {"thread_id": "test_session_2"}}
    query = "What skills are missing from my profile?"
    
    state = {
        "messages": [HumanMessage(content=query)],
        "active_resume_id": "REAL_0001",
        "active_jd_id": "JD_0001"
    }
    
    response = agent.invoke(state, config=config)
    
    messages = response["messages"]
    tool_calls = [m.tool_calls for m in messages if hasattr(m, 'tool_calls') and m.tool_calls]
    assert len(tool_calls) == 1
    assert tool_calls[0][0]["name"] == "get_missing_skills"
    
    final_message = messages[-1].content
    assert "According to the database:" in final_message
    assert "Missing Skills for Resume REAL_0001" in final_message

def test_agent_similar_resumes(agent):
    config = {"configurable": {"thread_id": "test_session_3"}}
    query = "Are there any similar peer resumes?"
    
    state = {
        "messages": [HumanMessage(content=query)],
        "active_resume_id": "REAL_0001",
        "active_jd_id": "JD_0001"
    }
    
    response = agent.invoke(state, config=config)
    
    messages = response["messages"]
    tool_calls = [m.tool_calls for m in messages if hasattr(m, 'tool_calls') and m.tool_calls]
    assert len(tool_calls) == 1
    assert tool_calls[0][0]["name"] == "get_similar_resumes"
    
    final_message = messages[-1].content
    assert "According to the database:" in final_message
    assert "Similar resumes in the same category" in final_message

def test_agent_rewrite_suggestion(agent):
    config = {"configurable": {"thread_id": "test_session_4"}}
    query = "Can you rewrite this: responsible for managing MySQL databases?"
    
    state = {
        "messages": [HumanMessage(content=query)],
        "active_resume_id": "REAL_0001",
        "active_jd_id": "JD_0001"
    }
    
    response = agent.invoke(state, config=config)
    
    messages = response["messages"]
    tool_calls = [m.tool_calls for m in messages if hasattr(m, 'tool_calls') and m.tool_calls]
    assert len(tool_calls) == 1
    assert tool_calls[0][0]["name"] == "rewrite_suggestion"
    
    final_message = messages[-1].content
    assert "According to the database:" in final_message
    assert "Original text:" in final_message
    assert "Rewritten Suggestion:" in final_message
