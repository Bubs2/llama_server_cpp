#include "llama_context.h"
#include "llama_model.h"
#include "llama_log.h"
#include "llama.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <format>
#include <ranges>

namespace llama_server::internal {

	// ===================================================================
	// LlamaContext::ContextDeleter
	// ===================================================================

	void LlamaContext::ContextDeleter::operator()(llama_context* context) const { llama_free(context); }

	// ===================================================================
	// LlamaContext
	// ===================================================================

	LlamaContext::LlamaContext(
		const llama_context_params& params,
		std::shared_ptr<LlamaModel> model
	) {
		model_ = model;

		context_ = std::unique_ptr<llama_context, ContextDeleter>(
			llama_init_from_model(model->get_data(), params)
		);
		if (!context_) {
			throw LlamaException("Failed to create llama context");
		}

		prefill_mask_ = std::vector<int8_t>(llama_n_batch(context_.get()), 0);
	}

	LlamaContext::~LlamaContext() {}

	LlamaContext::LlamaContext(LlamaContext&& other) noexcept = default;

	LlamaContext& LlamaContext::operator=(LlamaContext&& other) noexcept = default;

	bool LlamaContext::is_mtmd() const { return model_->get_mtmd() != nullptr; }

	const llama_vocab* LlamaContext::get_vocab() const { return model_->get_vocab(); }

	size_t LlamaContext::get_n_ctx() const { return llama_n_ctx(context_.get()); }

	size_t LlamaContext::get_n_vocab() const { return llama_vocab_n_tokens(model_->get_vocab()); }

	size_t LlamaContext::get_used_memory(llama_seq_id seq_id) const {
		return llama_memory_seq_pos_max(llama_get_memory(context_.get()), seq_id) + 1;
	}

	void LlamaContext::text_prefill(const llama_token* tokens, size_t n_tokens, bool logits_last) {
		const int32_t n_batch = llama_n_batch(context_.get());

		// It is guaranteed that text_tokens are not written by llama_decode.
		llama_token* casted_tokens = const_cast<llama_token*>(tokens);

		for (int32_t i = 0; i < n_tokens; i += n_batch) {
			int32_t n_eval = std::min(n_batch, (int32_t)n_tokens - i);

			prefill_mask_[n_eval - 1] = logits_last ? (i + n_eval) == n_tokens : 0;

			const int32_t ret = llama_decode(
				context_.get(),
				llama_batch{
					.n_tokens = n_eval,
					.token = casted_tokens + i,
					.embd = nullptr,
					.pos = nullptr,
					.n_seq_id = nullptr,
					.seq_id = nullptr,
					.logits = prefill_mask_.data()
				}
			);

			prefill_mask_[n_eval - 1] = 0;

			switch (ret) {
			case 0:
				break;
			case 1:
				throw LlamaException("Decoding failed: Could not find a KV slot for the batch.");
			case 2:
				throw ContextGenerateDirtyException("Decoding Aborted.");
			case -1:
				throw LlamaException("Decoding failed: Invalid input batch.");
			default:
				throw ContextGenerateDirtyException("Decoding failed: Unknown error");
			}
		}
	}

	void LlamaContext::text_prefill(std::span<llama_token> tokens, bool logits_last) {
		text_prefill(tokens.data(), tokens.size(), logits_last);
	}

	void LlamaContext::mtmd_prefill(std::span<IDChunksPtr const> chunks) {
		if (chunks.size() == 0) {
			log_info("No chunks to prefill");
			return;
		}

		for (size_t i = 0; i < chunks.size(); i++) {
			bool chunk_logits_last = (i + 1) == chunks.size();

			auto chunk_type = chunks[i]->type;
			if (chunk_type == TEXT) {
				eval_single_text_chunks(
					chunks[i],
					chunk_logits_last
				);
			}
			else if (chunk_type == IMAGE || chunk_type == AUDIO) {
				eval_single_mtmd_chunks(
					chunks[i],
					chunk_logits_last
				);
			}
			else {
				throw LlamaException(std::format("Unsupported chunk type: {}", static_cast<int>(chunk_type)));
			}
		}
	}

	void LlamaContext::step(llama_token token) {
		text_prefill(&token, 1, true);
	}

	void LlamaContext::KV_cleanup(int32_t head_keep, int32_t tail_spare, llama_seq_id seq_id) {
		llama_memory_t kv_mem = llama_get_memory(context_.get());
		if (!kv_mem) {
			throw LlamaException("Failed to get KV memory from context");
		}

		int32_t n_ctx = llama_n_ctx(context_.get());

		if (tail_spare == -1) {
			llama_memory_seq_rm(kv_mem, seq_id, head_keep, -1);
			return;
		}

		if (head_keep + tail_spare >= n_ctx) {
			log_warn("Prompt exceeds KV Cache.");
			llama_memory_seq_rm(kv_mem, seq_id, head_keep, -1);
			return;
		}

		llama_pos max_pos = llama_memory_seq_pos_max(kv_mem, seq_id);
		int32_t n_past = max_pos + 1;

		if (n_past + tail_spare <= n_ctx) return;

		int32_t n_discard = std::max(n_ctx >> 2, n_past + tail_spare - n_ctx);

		llama_pos p0 = head_keep;
		llama_pos p1 = head_keep + n_discard;

		if (p1 >= n_ctx) {
			log_warn("Too large system prompt, no memory available");
			llama_memory_seq_rm(kv_mem, seq_id, head_keep, -1);
			return;
		}

		if (p1 >= n_past) {
			llama_memory_seq_rm(kv_mem, seq_id, head_keep, -1);
			return;
		}

		llama_memory_seq_rm(kv_mem, seq_id, p0, p1);
		llama_memory_seq_add(kv_mem, seq_id, p1, n_past, -(llama_pos)n_discard);
	}

	void LlamaContext::eval_single_text_chunks(
		IDChunksPtr chunks,
		bool logits_last
	) {
		text_prefill(chunks->text_tokens, logits_last);
	}

	void LlamaContext::eval_single_mtmd_chunks(
		IDChunksPtr chunks,
		bool logits_last
	) {
		llama_pos new_n_past; // fake variable to satisfy the API
		if (mtmd_helper_eval_chunks(
			model_->get_mtmd(),
			context_.get(),
			chunks->chunks.get(),
			get_used_memory(),
			0,
			llama_n_batch(context_.get()),
			logits_last,
			&new_n_past)) { throw LlamaException("Multimodel decoding failed"); }
	}

}