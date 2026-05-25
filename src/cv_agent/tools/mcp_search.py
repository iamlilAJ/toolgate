"""This is an example of how to define a tool for CV Agent."""

import asyncio
import json
import logging
import textwrap
import time
from typing import TypedDict

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from cv_agent.core.registries import tool_registry
from cv_agent.tools.base import McpToolBase

_LOGGER = logging.getLogger(__name__)


class SearchInput(BaseModel):
    queries: list[str] = Field(description="a list of queries")
    filter_year: int = Field(description="filter out results older than this year")


class McpSearchResult(TypedDict):
    url: str
    title: str
    content: str
    engine: str
    publishedDate: str
    score: float


class SearchResult(TypedDict):
    url: str
    snippet: str


class McpSearchTool(McpToolBase):
    def __init__(
        self,
        mcp_server_url: str,
        engine: str,
        n_results: int,
        name: str = "search",
        description: str | None = None,
        args_schema: type[BaseModel] = SearchInput,
    ) -> None:
        if description is None:
            description = f"Conduct {engine} search by a list of queries."

        super().__init__(mcp_server_url, name, description, args_schema)

        self._engine = engine
        self._n = n_results

    async def invoke(self, queries: list[str], filter_year: int) -> list[SearchResult]:
        async with self._client as client:
            tasks = [
                client.call_tool("web_search", {"query": query, "engines": self._engine})
                for query in queries
            ]
            results = await asyncio.gather(*tasks)

        raw_results: list[McpSearchResult] = []
        for result in results:
            cnt = 0
            for content in result.content:
                if content.type != "text":
                    _LOGGER.warning(
                        "Encountered a search result whose type isn't text: %s", repr(content)
                    )
                    continue

                try:
                    raw_results.append(json.loads(content.text))
                except json.JSONDecodeError as e:
                    _LOGGER.warning(
                        "Error when decoding search result %s: %s", repr(content), repr(e)
                    )
                    continue

                cnt += 1
                if cnt >= self._n:
                    _LOGGER.info("Top-%d results stored; skip the rest.", cnt)
                    break

        if not raw_results:
            raise ValueError(f"No results not older than {filter_year} for queries {queries}.")

        return list(
            map(
                parse_mcp_search_result,
                filter(lambda res: filter_result_by_year(res, filter_year), raw_results),
            )
        )


@tool_registry.register("mcp_search")
def get_mcp_search_tool(
    server_url: str,
    engine: str,
    n_results: int,
    name: str = "search",
    description: str | None = None,
    args_schema: type[BaseModel] = SearchInput,
) -> StructuredTool:
    return McpSearchTool(
        server_url, engine, n_results, name, description, args_schema
    ).langchain_tool


def filter_result_by_year(result: McpSearchResult, year: int) -> bool:
    try:
        publish_year = time.strptime(result["publishedDate"], "%Y-%m-%dT%H:%M:%S").tm_year
        return publish_year >= year
    except ValueError as e:
        _LOGGER.warning("Error when comparing parsing published date: %s", e)
        return True


def parse_mcp_search_result(result: McpSearchResult) -> SearchResult:
    if result["engine"] == "wikisearch":
        snippet = textwrap.shorten(result["content"], 200)
    else:
        snippet = result["content"]

    return {"url": result["url"], "snippet": snippet}
