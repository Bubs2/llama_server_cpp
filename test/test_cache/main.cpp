#include "model_server.h"
#include "llama_configs.h"
#include "llama_session.h"
#include "history_manager.h"

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
		},
		"MiniCPM-V-4.5"
	);

	std::unique_ptr<LlamaSession> session = server.get_session(
		"MiniCPM-V-4.5",
		ContextConfig{
			.n_ctx = 512
		}
	);
	auto& history = session->access_history_manager();

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

		history.add_message(
			Message{
				.role = "user",
				.content = "正在进行KV Cache测试——测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试测试"
			}
		);

		so.is_first_token = true;
		so.start_time = std::chrono::high_resolution_clock::now();
		session->generate(
			GenConfig{
				.max_tokens = 256,
				.temperature = 0.4f,
				.top_k = 200,
				.penalty_last_n = 10,
				.penalty_repeat = 1.05f,
				.output_callback = so.cb
			}
		);
		history.add_message(
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