from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agents import AgentRegistry, PlannerAgent
from app.config import AppSettings
from app.errors import ErrorCode
from app.graph.state import GraphState
from app.observability import get_logger
from app.schemas import AgentExecutionContext, ErrorDetail, ReflectionTrace


logger = get_logger(__name__)


def build_graph(
    settings: AppSettings,
    planner: PlannerAgent,
    agent_registry: AgentRegistry,
):
    workflow = StateGraph(GraphState)

    async def planner_node(state: GraphState) -> dict[str, object]:
        attempt = len(state.get("planner_runs", [])) + 1
        plan, planner_trace = await planner.plan_with_trace(
            state["query"],
            hints=state.get("planner_hints"),
            available_agents=agent_registry.list_descriptors(),
            attempt=attempt,
        )
        logger.info(
            "planner_completed",
            extra={
                "plan_intent": plan.intent,
                "selected_agents": plan.agents,
                "planner_attempt": attempt,
            },
        )
        planner_runs = list(state.get("planner_runs", []))
        planner_runs.append(planner_trace)
        selected_agents = list(dict.fromkeys([*state.get("selected_agents", []), *plan.agents]))
        return {
            "plan": plan,
            "planner_runs": planner_runs,
            "selected_agents": selected_agents,
            "pending_agents": list(plan.agents),
        }

    async def route_node(state: GraphState) -> dict[str, object]:
        pending_agents = list(state.get("pending_agents", []))
        executed_agents = list(state.get("executed_agents", []))
        current_agent = pending_agents.pop(0) if pending_agents else ""
        if current_agent:
            executed_agents.append(current_agent)
        return {
            "current_agent": current_agent,
            "pending_agents": pending_agents,
            "executed_agents": executed_agents,
        }

    async def execute_agent_node(state: GraphState) -> dict[str, object]:
        current_agent = state.get("current_agent", "")
        result = await agent_registry.run(
            current_agent,
            AgentExecutionContext(
                query=state["query"],
                session_id=state["session_id"],
                request_id=state["request_id"],
                plan_intent=state.get("plan").intent if state.get("plan") else None,
                chat_history=state.get("chat_history", []),
                planner_hints=state.get("planner_hints", []),
            ),
        )
        intermediate = list(state.get("intermediate_results", []))
        intermediate.append(result.model_dump())
        agent_runs = list(state.get("agent_runs", []))
        agent_runs.append(result)
        tool_calls = list(state.get("tool_calls", []))
        tool_calls.extend(result.tool_calls)
        return {
            "intermediate_results": intermediate,
            "agent_runs": agent_runs,
            "tool_calls": tool_calls,
            "citations": result.citations or state.get("citations", []),
            "final_answer": result.answer if result.success else state.get("final_answer", ""),
        }

    async def reflect_node(state: GraphState) -> dict[str, object]:
        intermediate_results = state.get("intermediate_results", [])
        last_result = intermediate_results[-1] if intermediate_results else {}
        success = bool(last_result.get("success"))
        pending_agents = state.get("pending_agents", [])
        reflection_count = state.get("reflection_count", 0)
        reflections = list(state.get("reflections", []))
        agent_name = last_result.get("agent_name", "agent")
        metadata = last_result.get("metadata", {})
        reason = metadata.get("reason", last_result.get("error_code") or "execution_failed")

        if success and pending_agents:
            reflections.append(
                ReflectionTrace(
                    attempt=reflection_count + 1,
                    failed_agent=agent_name,
                    reason="agent_succeeded_with_pending_work",
                    action="route",
                )
            )
            return {"next_step": "route", "reflections": reflections}
        if success:
            reflections.append(
                ReflectionTrace(
                    attempt=reflection_count + 1,
                    failed_agent=agent_name,
                    reason="agent_succeeded",
                    action="finish",
                )
            )
            return {"next_step": "finish", "reflections": reflections}
        if reflection_count < settings.app.max_reflections:
            hints = list(state.get("planner_hints", []))
            added_hint = f"{agent_name}_failed:{reason}"
            hints.append(added_hint)
            reflections.append(
                ReflectionTrace(
                    attempt=reflection_count + 1,
                    failed_agent=agent_name,
                    reason=reason,
                    action="replan",
                    added_hint=added_hint,
                )
            )
            return {
                "reflection_count": reflection_count + 1,
                "planner_hints": hints,
                "reflections": reflections,
                "next_step": "planner",
            }
        reflections.append(
            ReflectionTrace(
                attempt=reflection_count + 1,
                failed_agent=agent_name,
                reason=reason,
                action="fallback",
                added_hint=f"{agent_name}_failed:{reason}",
            )
        )
        return {
            "error": ErrorDetail(
                code=ErrorCode.MAX_REFLECTIONS_REACHED,
                message="执行链路达到最大反思次数后仍未成功完成请求。",
                retryable=False,
            ),
            "pending_agents": ["fallback_agent"],
            "selected_agents": list(dict.fromkeys([*state.get("selected_agents", []), "fallback_agent"])),
            "reflections": reflections,
            "next_step": "route",
        }

    async def finish_node(state: GraphState) -> dict[str, object]:
        if state.get("final_answer"):
            return {}
        intermediate_results = state.get("intermediate_results", [])
        if not intermediate_results:
            return {
                "final_answer": "系统没有产出有效结果。",
                "error": ErrorDetail(
                    code=ErrorCode.EMPTY_RESULT,
                    message="系统没有产出有效结果。",
                    retryable=False,
                ),
            }
        return {"final_answer": intermediate_results[-1].get("answer", "")}

    workflow.add_node("planner", planner_node)
    workflow.add_node("route", route_node)
    workflow.add_node("execute_agent", execute_agent_node)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("finish", finish_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "route")
    workflow.add_conditional_edges(
        "route",
        lambda state: "execute_agent" if state.get("current_agent", "") else "finish",
        {
            "execute_agent": "execute_agent",
            "finish": "finish",
        },
    )
    workflow.add_edge("execute_agent", "reflect")
    workflow.add_conditional_edges(
        "reflect",
        lambda state: state.get("next_step", "finish"),
        {
            "route": "route",
            "planner": "planner",
            "finish": "finish",
        },
    )
    workflow.add_edge("finish", END)

    return workflow.compile()
