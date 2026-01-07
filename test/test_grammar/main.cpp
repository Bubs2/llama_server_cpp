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
			.n_ctx = 10240,
			.n_batch = 2048,
			.n_ubatch = 2048,
		}
		);

	session->set_token_estimate_strategy(TokenEstimateStrategy(
		[](const TokenizeCallback& tokenize_cb, std::string_view str) {
		return (str.size() + 1) / 2;
	},
		64
	));

	std::vector<Message> head_msgs;
	std::vector<Message> tail_msgs;
	std::vector<Tool> tools;

	struct simple_output {
		std::string response_buffer;

		OutputCallback cb = [&](std::string&& text) {
			response_buffer += text;
			std::cout << text;
			return true;
		};
	};
	simple_output so;

	Grammar simple_grammar = Grammar{
		.value = R"(
			root   ::= object
			object ::= "{" ws "\"action\"" ":" ws action-enum "," ws "\"target\"" ":" ws target-val ws "}"
			ws     ::= [ \t\n]*
			string ::= "\"" [^"]* "\""

			action-enum ::= "\"move\"" | "\"jump\"" | "\"attack\"" | "\"mine\"" | "\"craft\""
			target-val  ::= "\"" [a-zA-Z0-9_-]+ "\""
		)"
	};

	tail_msgs.emplace_back(
		Message{
			.role = "user",
			.content = "攻击Zombie-121。"
		}
	);
	session->generate(
		head_msgs, tail_msgs, tools,
		GenConfig{
			.enable_thinking = false,
			.temperature = 0.4f,
			.top_k = 40,
			.penalty_last_n = 64,
			.penalty_repeat = 1.15f,
			.grammar = simple_grammar,
			.output_callback = so.cb,
		}
	);

	std::cout << std::endl;

	tail_msgs[0].content = "移动到Village-4Y2。";
	session->generate(
		head_msgs, tail_msgs, tools,
		GenConfig{
			.enable_thinking = false,
			.temperature = 0.4f,
			.top_k = 40,
			.penalty_last_n = 64,
			.penalty_repeat = 1.15f,
			.grammar = simple_grammar,
			.output_callback = so.cb,
		}
	);

	std::cout << std::endl;

	server.shutdown();

	system("pause");
}