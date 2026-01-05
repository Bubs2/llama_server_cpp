#pragma once

#include "llama_configs.h"
#include "llama_strategy.h"
#include "llama_exception.h"
#include "llama_inputs.h"

#include <memory>

namespace llama_server {

	namespace internal {
		class LlamaModel;
		class LlamaContext;
		class InputEncoder;
		class KVScheduler;
		class Tokenizer;
		class Templater;
		class Sampler;
		class Streamer;
	}
	

	class LlamaSession {
	public:
		LlamaSession(
			ContextConfig context_config,
			std::shared_ptr<internal::LlamaModel> model
		);
		~LlamaSession();

		LlamaSession(const LlamaSession&) = delete;
		LlamaSession& operator=(const LlamaSession&) = delete;
		LlamaSession(LlamaSession&&) noexcept;
		LlamaSession& operator=(LlamaSession&&) noexcept;

		void generate(
			std::vector<Message> head_msgs,
			std::vector<Message> tail_msgs,
			std::vector<Tool> tools,
			const GenConfig& gen_config
		);

		void set_token_estimate_strategy(TokenEstimateStrategy&& strategy);
	private:
		std::unique_ptr<internal::LlamaContext> context_;

		std::unique_ptr<internal::Tokenizer> tokenizer_;
		std::unique_ptr<internal::Templater> templater_;

        std::unique_ptr<internal::InputEncoder> input_encoder_;
		std::unique_ptr<internal::KVScheduler> kv_scheduler_;

		std::unique_ptr<internal::Sampler> sampler_;
		std::unique_ptr<internal::Streamer> streamer_;
	};

}