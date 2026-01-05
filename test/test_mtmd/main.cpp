#include "model_server.h"
#include "llama_configs.h"
#include "llama_inputs.h"
#include "llama_session.h"

#include <windows.h>
#include <memory>
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
			.n_gpu_layers = 99,
			.mtmd_path = "D:/CraftTools/AI/my_ai_assistant/model/mmproj-model-f16.gguf",
		},
		"MiniCPM-V-4.5"
	);

	std::unique_ptr<LlamaSession> session = server.get_session(
		"MiniCPM-V-4.5",
		ContextConfig{
			.n_ctx = 10240,
			.n_batch = 2048,
			.n_ubatch = 2048,
		}
	);
	
	std::vector<Message> head_msgs;
	std::vector<Message> tail_msgs;
	std::vector<Tool> tools;

	tail_msgs.emplace_back(
		Message{
			.role = "user",
			.content = "这里有一张图片：<__path:./image/prac_back.png__>"
		}
	);

	struct simple_output {
		std::string response_buffer;

		OutputCallback cb = [&](std::string&& text) {
			response_buffer += text;
			std::cout << text;

			return true;
		};
	};
	simple_output so;
	while (true) {
		std::string input;
		std::getline(std::cin, input);

		so.response_buffer = std::string();

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
	}

	server.shutdown();

	system("pause");
}