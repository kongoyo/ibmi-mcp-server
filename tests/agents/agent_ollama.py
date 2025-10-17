import anyio
import asyncio
from agno.agent import Agent
from agno.tools.mcp import MCPTools
from dotenv import load_dotenv
import argparse
from utils import get_model


load_dotenv(override=True)
url = "http://127.0.0.1:3010/mcp"

# Default model_id is set to granite-code:8b
async def main(prompt=None, dry_run=False, model_id="granite-code:8b"):
    async with MCPTools(url=url, transport="streamable-http") as tools:
        # Print available tools for debugging
        result = await tools.session.list_tools()
        tools_list = result.tools  # Extract the tools list from the result

        # Create agent with all tools but instruct it to prefer security tools
        if not dry_run:
            agent = Agent(
                model=get_model(model_id),
                tools=[tools],  # Use original tools but with specific instructions
                name="agno-agent",
                description=f"An agent that specializes in IBM i performance analysis.",
                show_tool_calls=True,
                debug_mode=True,
                debug_level=1,
                markdown=True,
                additional_context={
                    "tool_annotations": {
                        tool.name: tool.annotations
                        for tool in tools_list
                        if tool.annotations
                    }
                },
            )

            # Use provided prompt or default prompt
            user_prompt = prompt if prompt else "我的系統CPU平均使用率是多少?"

            await agent.aprint_response(user_prompt, stream=False)


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="IBM i MCP Agent Test - Query your IBM i system using natural language"
    )
    parser.add_argument("-p", "--prompt", type=str, help="Prompt to send to the agent")
    parser.add_argument(
        "-d", "--dry-run", action="store_true", help="Run in dry mode without executing actions",
    )

    parser.add_argument(
        "--model-id",
        type = str,
        default="granite-code:8b",
        help="Model ID to use for the agent (default: ollama:qwen2.5:latest)",
    )

    args = parser.parse_args()

    asyncio.run(main(args.prompt, args.dry_run, args.model_id))
