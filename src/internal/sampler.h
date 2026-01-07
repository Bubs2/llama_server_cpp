#pragma once

#include "llama_exception.h"
#include "llama_configs.h"
#include "llama.h"
#include "chat.h"

#include <memory>
#include <vector>

struct common_chat_params;

namespace llama_server::internal {

	class LlamaContext;

	class Sampler {
	public:
		Sampler(const LlamaContext& context);
		~Sampler();

		void set(
			const GenConfig& gen_config,
			const Grammar& auto_grammar
		);

		llama_token apply();
	private:
		struct SamplerDeleter {
			void operator()(llama_sampler* sampler) const;
		};
		using SamplerPtr = std::unique_ptr<llama_sampler, SamplerDeleter>;

		SamplerPtr ptr_ = nullptr;

		const LlamaContext& context_;

		std::vector<llama_token_data> candidates_buffer_;

		static llama_sampler* get_grammar_sampler(const Grammar& grammar, const llama_vocab* vocab);
	};

}