#pragma once

#include "llama_exception.h"
#include "llama_strategy.h"
#include "id_chunk.h"
#include "chat.h"

#include <memory>
#include <vector>
#include <string_view>
#include <unordered_set>

namespace llama_server {
	class TokenEstimateStrategy;
}

namespace llama_server::internal {

	class LlamaContext;
	class Templater;
    class Tokenizer;

	namespace input_encoder_detail {
		class MediaHelper;
	}

	class InputEncoder {
	public:
		InputEncoder(
			const LlamaContext& context,
			const Templater& templater,
			const Tokenizer& tokenizer,
			TokenEstimateStrategy&& token_estimate_strategy
		);
		~InputEncoder();

		std::vector<IDChunksPtr> operator()(
			std::vector<common_chat_msg>&& head_msgs,
			std::vector<common_chat_msg>&& tail_msgs,
			std::vector<common_chat_tool>&& tools,
			size_t max_tokens
		);

		size_t get_used_messages_cache();
		common_chat_params get_chat_params_cache();

		void set_token_estimate_strategy(TokenEstimateStrategy&& strategy) { token_estimate_strategy_ = std::move(strategy); }
	private:
        const LlamaContext& context_;
		const Templater& templater_;
		const Tokenizer& tokenizer_;
		std::unique_ptr<input_encoder_detail::MediaHelper> media_helper_;

		TokenEstimateStrategy token_estimate_strategy_;

		std::unordered_set<IDChunksPtr, IDChunksPtrHash, IDChunksPtrEqual> image_chunks_cache_;

		size_t used_messages_cache_ = 0;
		common_chat_params chat_params_cache_;

		size_t estimate_text_tokens(std::string_view str);
		size_t estimate_mtmd_tokens(std::string_view str);
		common_chat_templates_inputs prune_with_precache(
			std::vector<common_chat_msg>&& head_msgs,
			std::vector<common_chat_msg>&& tail_msgs,
			std::vector<common_chat_tool>&& tools,
			size_t max_tokens
		);
	};

}