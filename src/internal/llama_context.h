#pragma once

#include "llama_exception.h"
#include "id_chunk.h"
#include "llama.h"
#include "mtmd.h"

#include <memory>
#include <vector>
#include <span>

namespace llama_server::internal {

	class LlamaModel;

	class LlamaContext {
	public:
		LlamaContext(
			const llama_context_params& params,
			std::shared_ptr<LlamaModel> model
		);
		~LlamaContext();

		LlamaContext(const LlamaContext&) = delete;
		LlamaContext& operator=(const LlamaContext&) = delete;
		LlamaContext(LlamaContext&&) noexcept;
		LlamaContext& operator=(LlamaContext&&) noexcept;

		llama_context* get_data() const { return context_.get(); }
		std::shared_ptr<LlamaModel> get_model() const { return model_; }
		const llama_vocab* get_vocab() const;

		size_t get_n_ctx() const;
		size_t get_n_vocab() const;
		size_t get_used_memory(llama_seq_id seq_id = 0) const;

		void text_prefill(const llama_token* tokens, size_t n_tokens, bool logits_last = false);
		void text_prefill(std::span<llama_token> tokens, bool logits_last = false);
		void mtmd_prefill(std::span<IDChunkPtr const> chunks);
		void step(llama_token token);

		void KV_cleanup(
			int32_t head_keep = 0,
			int32_t tail_spare = -1,
			llama_seq_id seq_id = 0
		);

	private:
		struct ContextDeleter {
			void operator()(llama_context* context) const;
		};

		std::unique_ptr<llama_context, ContextDeleter> context_;
		std::shared_ptr<LlamaModel> model_;

		std::vector<int8_t> prefill_mask_;

		void eval_single_text_chunk(
			IDChunkPtr chunk,
			bool logits_last
		);
		void eval_single_mtmd_chunk(
			IDChunkPtr chunk
		);
	};

	class ContextGenerateDirtyException : public LlamaException {
	public:
		using LlamaException::LlamaException;
	};

}