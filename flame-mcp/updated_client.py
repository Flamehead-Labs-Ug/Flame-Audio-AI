import asyncio
import sys
import traceback
from urllib.parse import urlparse

from mcp import ClientSession
from mcp.client.sse import sse_client

def print_items(name: str, result: any) -> None:
    print(f"\nAvailable {name}:")
    items = getattr(result, name)
    if items:
        for item in items:
            print(" *", item)
    else:
        print("No items available")

async def main(server_url: str, article_url: str):
    """
    Connect to the MCP server and call the summarize_wikipedia_article tool.
    
    Args:
        server_url: Full URL to SSE endpoint (e.g. http://localhost:8000/sse)
        article_url: Wikipedia URL to fetch and summarize
    """
    if urlparse(server_url).scheme not in ("http", "https"):
        print("Error: Server URL must start with http:// or https://")
        sys.exit(1)
        
    try:
        async with sse_client(server_url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                print("Connected to MCP server at", server_url)
                print_items("tools", await session.list_tools())
                print_items("resources", await session.list_resources())
                print_items("prompts", await session.list_prompts())
                
                print("\nCalling summarize_wikipedia_article tool...")
                response = await session.call_tool("summarize_wikipedia_article", arguments={"url": article_url})
                print("\n=== Summarized Wikipedia Article ===\n")
                print(response)
    except Exception as e:
        print(f"Error connecting to server: {e}")
        traceback.print_exception(type(e), e, e.__traceback__)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: uv run -- client.py <server_url> <wikipedia_article_url>\n"
            "Example: uv run -- client.py $MCP_URL/sse https://en.wikipedia.org/wiki/India"
        )
        sys.exit(1)
    
    server_url = sys.argv[1]
    article_url = sys.argv[2]
    asyncio.run(main(server_url, article_url))
