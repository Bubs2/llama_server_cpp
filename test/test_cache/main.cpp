#include "model_server.h"
#include "llama_configs.h"
#include "llama_strategy.h"
#include "llama_inputs.h"
#include "llama_session.h"

#include <windows.h>
#include <memory>
#include <iostream>
#include <consoleapi2.h>
#include <nlohmann/json.hpp>
#include <chrono>

using namespace llama_server;
using json = nlohmann::json;

int main(int argc, char* argv[]) {
	SetConsoleOutputCP(65001);
	SetConsoleCP(65001);

	ModelServer& server = ModelServer::get_server();

	server.load_model(
		ModelConfig{
			.model_path = "D:/CraftTools/AI/my_ai_assistant/model/MiniCPM-V-4_5-Q4_K_M.gguf",
			.n_gpu_layers = 99,
		},
		"MiniCPM-V-4.5"
	);

	std::unique_ptr<LlamaSession> session = server.get_session(
		"MiniCPM-V-4.5",
		ContextConfig{
			.n_ctx = 2048,
			.n_batch = 2048,
			.n_ubatch = 2048,
		}
	);

	// Accurate estimation(default) 700-1000ms TTFT when full ctx : Simple estimation 200-300ms TTFT when full ctx
	session->set_token_estimate_strategy(TokenEstimateStrategy(
		[](const TokenizeCallback& tokenize_cb, std::string_view str) {
			// Simple estimation: 1 token per 2 characters
			return (str.size() + 1) / 2;
		},
		64 // Margin
	));

	std::vector<Message> head_msgs;
	std::vector<Message> tail_msgs;
	std::vector<Tool> tools;

	struct simple_output {
		std::string response_buffer;

		std::chrono::high_resolution_clock::time_point start_time;
		bool is_first_token = true;

		OutputCallback cb = [&](std::string&& text) {
			if (is_first_token) {
				auto now = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

				std::cout << " [TTFT: " << duration << "ms] ";

				is_first_token = false;
			}

			response_buffer += text;
			std::cout << text;

			return true;
		};
	};
	simple_output so;

	for (int i = 0; i < 100; ++i) {
		so.response_buffer = std::string();

		tail_msgs.emplace_back(
			Message{
				.role = "user",
				.content = "正在进行KV Cache测试——测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试"
			}
		);

		so.is_first_token = true;
		so.start_time = std::chrono::high_resolution_clock::now();
		session->generate(
			head_msgs, tail_msgs, tools,
			GenConfig{
				.max_tokens = 256,
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
	}

	server.shutdown();

	system("pause");
}