import * as readline from "readline/promises";
import Groq from "groq-sdk";
import { tavily, type TavilySearchResponse } from "@tavily/core";
import { ChatCompletionMessageParam } from "groq-sdk/resources/chat/completions";

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY || "" });
const tvly = tavily({ apiKey: process.env.TAVILY_API_KEY || "" });

const messages: ChatCompletionMessageParam[] = [
  {
    role: "system",
    content: `You are a smart helpful assistant who answers the asked questions. You have access to the following tool:
      
1. webSearch - Search the latest information and real time data on the internet.
      
When you need to use the webSearch tool, respond with a JSON object containing:
- "query": The search query string

Important: Always use the tool calling format specified by the API. Do not use XML or other formats.`,
  },
];

async function main() {
  const rl = await readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  console.log("Welcome! Type 'bye' to exit.\n");

  while (true) {
    const question = await rl.question("You: ");

    if (question.toLowerCase() === "bye") {
      console.log("Goodbye!");
      break;
    }

    messages.push({
      role: "user",
      content: question,
    });

    try {
      while (true) {
        const completions = await groq.chat.completions.create({
          messages: messages,
          model: "llama-3.3-70b-versatile",
          temperature: 0,
          tools: [
            {
              type: "function",
              function: {
                name: "webSearch",
                description:
                  "Search the latest information and real time data on the internet.",
                parameters: {
                  type: "object",
                  properties: {
                    query: {
                      type: "string",
                      description: "The web search query",
                    },
                  },
                  required: ["query"],
                },
              },
            },
          ],
          tool_choice: "auto",
        });

        const choice = completions.choices?.[0];
        if (!choice) {
          console.error(
            "No choices returned by groq.chat.completions.create()"
          );
          break;
        }

        const message = choice.message;

        // If there's a content response without tool calls, display it
        if (message.content) {
          console.log("\nAssistant:", message.content);
          messages.push({
            role: "assistant",
            content: message.content,
          });
          break;
        }

        // Check for tool calls
        const toolCalls = message.tool_calls;

        if (!toolCalls || toolCalls.length === 0) {
          // No tool calls and no content - add the assistant message anyway
          if (message.content) {
            messages.push({
              role: "assistant",
              content: message.content,
            });
          }
          break;
        }

        // Process tool calls
        console.log("\nSearching for information...");

        for (const toolCall of toolCalls) {
          if (toolCall.function.name === "webSearch") {
            try {
              const args = JSON.parse(toolCall.function.arguments);
              const toolResult = await webSearch(args);

              // Add the tool call to messages
              messages.push({
                role: "assistant",
                content: null,
                tool_calls: [toolCall],
              });

              // Add the tool result to messages
              messages.push({
                role: "tool",
                tool_call_id: toolCall.id,
                content: toolResult,
              });
            } catch (error) {
              console.error("Error parsing tool arguments:", error);
              messages.push({
                role: "tool",
                tool_call_id: toolCall.id,
                content: "Error: Invalid search parameters",
              });
            }
          }
        }

        // Now get the final response with the search results
        const finalCompletion = await groq.chat.completions.create({
          messages: messages,
          model: "llama-3.3-70b-versatile",
          temperature: 0,
        });

        const finalChoice = finalCompletion.choices?.[0];
        if (finalChoice?.message?.content) {
          console.log("\nAssistant:", finalChoice.message.content);
          messages.push({
            role: "assistant",
            content: finalChoice.message.content,
          });
          break;
        }
      }
    } catch (error: any) {
      console.error("\nError:", error?.message || "An error occurred");

      // If it's a tool calling error, try again without tools
      if (error?.status === 400 && error?.error?.code === "tool_use_failed") {
        console.log("Retrying without tool calls...");

        // Remove the last user message and try a regular completion
        messages.pop(); // Remove user message
        messages.push({
          role: "user",
          content: question,
        });

        const fallbackCompletion = await groq.chat.completions.create({
          messages: messages,
          model: "llama-3.3-70b-versatile",
          temperature: 0,
          // Don't include tools parameter
        });

        if (fallbackCompletion.choices[0]?.message?.content) {
          console.log(
            "\nAssistant:",
            fallbackCompletion.choices[0].message.content
          );
          messages.push({
            role: "assistant",
            content: fallbackCompletion.choices[0].message.content,
          });
        }
      }
    }
  }

  rl.close();
}

async function webSearch({ query }: { query: string }): Promise<string> {
  if (!query || !query.trim()) {
    return "Error: No search query provided";
  }

  try {
    console.log(`Searching for: "${query}"`);
    const response: TavilySearchResponse = await tvly.search(query, {
      maxResults: 5,
      includeAnswer: false,
      includeRawContent: false,
    });

    if (response.results && response.results.length > 0) {
      const formattedResults = response.results
        .slice(0, 3) // Limit to top 3 results
        .map((result, index) => {
          let content = result.content || "";
          // Truncate content if too long
          if (content.length > 500) {
            content = content.substring(0, 500) + "...";
          }
          return `[${index + 1}] Title: ${result.title || "No title"}\nURL: ${
            result.url
          }\nContent: ${content}`;
        })
        .join("\n\n");

      return `Search Results for "${query}":\n\n${formattedResults}\n\nTotal results found: ${response.results.length}`;
    }

    return `No results found for "${query}". Please try a different search query.`;
  } catch (error) {
    console.error("Error in webSearch:", error);
    return `Error occurred while searching for "${query}". Please try again.`;
  }
}

main().catch(console.error);
