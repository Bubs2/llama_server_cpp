#include "model_server.h"
#include "llama_configs.h"
#include "llama_inputs.h"
#include "llama_session.h"

#include <windows.h>
#include <memory>
#include <string_view>
#include <iostream>
#include <consoleapi2.h>
#include <nlohmann/json.hpp>

using namespace llama_server;
using json = nlohmann::json;

int main(int argc, char* argv[]) {
	SetConsoleOutputCP(65001);
	SetConsoleCP(65001);

	ModelServer& server = ModelServer::get_server();

	server.load_model(
		ModelConfig{
			.model_path = "D:/CraftTools/AI/my_ai_assistant/model/MiniCPM-V-4_5-Q4_K_M.gguf",
			.mtmd_path = "D:/CraftTools/AI/my_ai_assistant/model/mmproj-model-f16.gguf"
		},
		"MiniCPM-V-4.5"
	);

	std::unique_ptr<LlamaSession> session = server.get_session(
		"MiniCPM-V-4.5",
		ContextConfig{
			.n_ctx = 10240
		}
	);

	std::vector<Message> head_msgs;
	std::vector<Message> tail_msgs;
	std::vector<Tool> tools;

	auto tool_callback = [](std::string_view parameters) {
        json j = json::parse(parameters);
		int num_1 = j["arguments"]["num_1"];
		int num_2 = j["arguments"]["num_2"];
		return std::format("计算器结果：{}", num_1 + num_2);
	};
	tools.emplace_back(Tool{
		.name = "Sum",
		.description = "对两个整数进行求和运算。对所有两数相加整数，直接用参数调用Sum。",
		.parameters = R"(
			{
				"type": "object",
				"properties" : {
					"num_1": {
						"type": "integer",
						"description" : "加数1。"
					},
					"num_2": {
						"type": "integer",
						"description" : "加数2。"
					}
				},
				"required": ["num_1", "num_2"]
			}
		)"
	});

	struct simple_output {
		std::string response_buffer;

		std::string tool_start_pattern = "<tool_call>";
		std::string tool_end_pattern = "</tool_call>";
		bool tool_calling = false;
		std::string tool_call_buffer;
		std::string tool_result_buffer;

		ToolCallback tcb;

		OutputCallback cb = [&](std::string&& text) {
			response_buffer += text;
			std::cout << text;

			if (tool_calling) {
				size_t end_pos = text.find(tool_end_pattern);
				if (end_pos != std::string::npos) {
					tool_call_buffer += text.substr(0, end_pos);
					tool_result_buffer = tcb(tool_call_buffer);
					return false;
				}
				tool_call_buffer += text;
				return true;
			}

			size_t start_pos = text.find(tool_start_pattern);
			if (start_pos != std::string::npos) {
				tool_calling = true;
				tool_call_buffer.clear();
				tool_call_buffer += text.substr(start_pos + tool_start_pattern.size());
				return true;
			}

			return true;
		};
	};
    simple_output so;
	so.tcb = tool_callback;
	while (true) {
		std::string input;
		std::getline(std::cin, input);

		so.response_buffer = std::string();
		so.tool_result_buffer = std::string();
		so.tool_call_buffer = std::string();

		tail_msgs.emplace_back(
			Message{
				.role = "user",
				.content = std::move(input)
			}
		);
		session->generate(
			head_msgs, tail_msgs, tools,
			GenConfig{
				.temperature = 0.4f,
				.top_k = 200,
				.penalty_last_n = 10,
				.penalty_repeat = 1.05f,
				.output_callback = so.cb
			}
		);
		tail_msgs.emplace_back(
			Message{
				.role = "assistant",
				.content = std::move(so.response_buffer)
			}
		);

		std::cout << std::endl;

		if (so.tool_result_buffer.empty()) continue;

		so.response_buffer = std::string();

		tail_msgs.emplace_back(
			Message{
				.role = "tool",
				.content = std::move(so.tool_result_buffer)
			}
		);
		session->generate(
			head_msgs, tail_msgs, tools,
			GenConfig{
				.temperature = 0.2f,
				.top_k = 100,
				.output_callback = so.cb
			}
		);
		tail_msgs.emplace_back(
			Message{
				.role = "assistant",
				.content = std::move(so.response_buffer)
			}
		);

		std::cout << std::endl;
	}

	server.shutdown();

	system("pause");
}