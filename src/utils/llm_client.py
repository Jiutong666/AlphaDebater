from __future__ import annotations

"""
LLM 客户端封装模块。

基于 OpenAI 兼容接口，默认对接 DeepSeek API。
提供统一的聊天补全调用 + 自动重试机制。
"""

import json
import time
from typing import Any, Callable

from openai import OpenAI

from src.config.settings import settings


class LLMClient:
    """OpenAI 兼容的 LLM 调用客户端。

    封装了 API 密钥管理、超时重试、异常处理等生产级关注点。

    Constructor parameters override settings; useful for testing.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize LLM client. Falls back to global settings for any `None` arg."""
        self.client = OpenAI(
            api_key=api_key if api_key is not None else settings.llm_api_key,
            base_url=base_url if base_url is not None else settings.llm_base_url,
        )
        self.model = model if model is not None else settings.llm_model
        self.temperature = temperature if temperature is not None else settings.temperature
        self.max_tokens = max_tokens if max_tokens is not None else settings.max_tokens

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        *,
        prefill: str = "",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> str:
        """发送聊天请求并返回模型回复文本。

        内置指数退避重试，应对网络抖动或速率限制。

        Args:
            system_prompt: 系统角色提示词，定义 Agent 人设与行为。
            user_message:  用户消息，包含待分析的数据与辩论上下文。
            prefill:       可选的 assistant prefill 文本，锁定模型开局语气。
                           模型将从 prefill 结束处继续生成。
            max_retries:   最大重试次数。
            retry_delay:   重试间隔（秒），每次重试翻倍。

        Returns:
            模型生成的文本回复。

        Raises:
            RuntimeError: 当所有重试均失败时抛出。
        """
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                messages: list[dict[str, str]] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ]
                if prefill:
                    messages.append({"role": "assistant", "content": prefill})

                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=messages,
                )
                content: str = response.choices[0].message.content or ""
                return content

            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    wait: float = retry_delay * (2 ** attempt)
                    time.sleep(wait)
                continue

        raise RuntimeError(
            f"LLM 调用失败，已重试 {max_retries} 次。"
            f"最后错误: {last_error}"
        )


    def chat_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[dict[str, Any]],
        tool_registry: dict[str, Callable[..., dict[str, Any]]],
        *,
        prefill: str = "",
        max_tool_turns: int = 3,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> tuple[str, list[dict[str, Any]]]:
        """带工具调用的聊天方法 — Neuro-Symbolic 计算核心。

        处理完整的工具调用循环：
        1. 发送消息 + 工具定义 → LLM
        2. 若 LLM 返回 tool_calls → 执行工具 → 将结果反馈 → 回到步骤 1
        3. 若 LLM 返回纯文本 → 返回 (文本, 工具调用日志)

        工具调用日志记录每次调用的工具名、参数和返回值，
        可用于调试和透明度展示。

        Args:
            system_prompt:  系统角色提示词。
            user_message:   用户消息。
            tools:          OpenAI 兼容的工具定义列表。
            tool_registry:  工具名 → 可调用函数的映射。
            prefill:        可选的 assistant prefill 文本。
            max_tool_turns: 最大工具调用往返次数 (防无限循环)。
            max_retries:    API 调用最大重试次数。
            retry_delay:    重试间隔（秒），每次翻倍。

        Returns:
            (final_text_response, tool_call_log)
        """
        # prefill 不再作为独立 assistant 消息（会与 tool_calls 产生冲突）。
        # 改为注入到 user_message 末尾作为输出格式要求。
        user_message_with_prefill: str = user_message
        if prefill:
            user_message_with_prefill += (
                f"\n\n## 输出格式硬性要求\n"
                f"你的回复必须以以下文字作为开头（直接续写，不要重复该文字）：\n"
                f"```\n{prefill}\n```\n"
                f"如果需要先调用工具计算，在工具返回结果后，"
                f"请以上述文字开头输出最终回复。"
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message_with_prefill},
        ]

        tool_call_log: list[dict[str, Any]] = []

        for turn in range(max_tool_turns):
            response = self._api_call_with_retry(
                messages, tools, max_retries, retry_delay
            )

            if response is None:
                return ("⚠️ LLM API 调用失败，所有重试已耗尽。", tool_call_log)

            choice = response.choices[0]
            message = choice.message

            # 检查是否有 tool_calls
            if message.tool_calls:
                # 构建 assistant 消息（处理 content 可能为 None 的情况）
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
                # DeepSeek 兼容: 仅当 content 非空时才包含 content 字段
                if message.content:
                    assistant_msg["content"] = message.content
                messages.append(assistant_msg)

                # 执行每个工具调用
                for tc in message.tool_calls:
                    tool_name = tc.function.name
                    try:
                        tool_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        tool_result = {"error": f"无法解析工具参数: {tc.function.arguments}"}
                    else:
                        fn = tool_registry.get(tool_name)
                        if fn is None:
                            tool_result = {"error": f"未知工具: {tool_name}"}
                        else:
                            try:
                                tool_result = fn(**tool_args)
                            except Exception as exc:
                                tool_result = {"error": f"工具执行失败: {exc}"}

                    tool_call_log.append({
                        "tool": tool_name,
                        "arguments": tc.function.arguments,
                        "result": tool_result,
                    })

                    # 将工具结果加入消息历史
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    })

                # 每次工具返回后立即追加停止指令
                stop_msg = (
                    "工具计算结果已返回。请**立即**输出你的完整辩论发言正文，"
                    "不得再次调用任何工具。"
                )
                if prefill:
                    stop_msg += " 回复必须以指定的 markdown 标题开头。"
                messages.append({"role": "user", "content": stop_msg})
                continue

            # 没有 tool_calls — 可能是绕过工具的纯文本
            # 第一轮：强制要求先调用计算工具，防止在文本中编造数字
            if turn == 0:
                assistant_msg = {"role": "assistant", "content": message.content or ""}
                messages.append(assistant_msg)
                force_tool_msg = (
                    "请**先调用必要的计算工具**（如 calculate_target_price、"
                    "calculate_upside_downside 等），获取精确计算结果后，"
                    "再输出完整的辩论正文。禁止在未调用工具的情况下直接在文本中推算数字。"
                )
                messages.append({"role": "user", "content": force_tool_msg})
                continue

            # 后续轮次 — 允许纯文本返回
            content: str = message.content or ""
            if prefill and content and not content.startswith(prefill[:30]):
                content = prefill + content
            return (content, tool_call_log)

        # 所有轮次耗尽后仍未产出文本 → 补一次不带 tools 的最终调用
        final_response = self._api_call_with_retry(
            messages, tools=None, max_retries=max_retries, retry_delay=retry_delay
        )
        if final_response is not None:
            content = final_response.choices[0].message.content or ""
            if prefill and content and not content.startswith(prefill[:30]):
                content = prefill + content
            if content:
                return (content, tool_call_log)

        return (
            "⚠️ 工具调用达到最大轮次限制，未能获得最终回复。",
            tool_call_log,
        )

    def _api_call_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_retries: int,
        retry_delay: float,
    ) -> Any:
        """内部 API 调用，带指数退避重试。

        Args:
            messages:    完整的消息列表。
            tools:       工具定义列表，为 None 时不启用工具。
            max_retries: 最大重试次数。
            retry_delay: 基础重试延迟（秒）。

        Returns:
            API 响应对象，或 None（所有重试失败时）。
        """
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "messages": messages,
                }
                if tools:
                    kwargs["tools"] = tools
                return self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    wait: float = retry_delay * (2 ** attempt)
                    time.sleep(wait)
                continue

        return None


# 全局 LLM 客户端单例
llm_client = LLMClient()
