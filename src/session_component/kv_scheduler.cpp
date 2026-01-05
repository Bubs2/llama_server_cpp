#include "kv_scheduler.h"
#include "llama_log.h"
#include "llama_context.h"
#include "llama.h"

#include <ranges>

namespace llama_server::internal {

	KVScheduler::KVScheduler(LlamaContext& context)
		: context_(context) {
	}

	void KVScheduler::prefill_text_cache(std::vector<llama_token> tokens) {
		size_t keep = 0;
		while (keep < tokens.size() && keep < prev_tokens_.size() && tokens[keep] == prev_tokens_[keep]) ++keep;

		context_.KV_cleanup(keep);

		std::vector<llama_token> prefill_tokens(tokens.begin() + keep, tokens.end());
		try {
			context_.text_prefill(prefill_tokens);
		}
		catch (const LlamaException& e) {
			prev_tokens_.clear(); // Reset to prevent any unpredictable state;
			throw LlamaException("Error prefilling cache: " + std::string(e.what()));
		}

		prev_tokens_ = std::move(tokens);
	}

	void KVScheduler::prefill_mtmd_cache(std::span<IDChunksPtr const> chunks) {
		size_t perfect_keep = 0;
		size_t last_keep = 0;
		size_t kept_chunks = 0;

		while (kept_chunks < chunks.size() && kept_chunks < prev_chunks_info_.size()) {
			auto& chunk = chunks[kept_chunks];
			auto chunk_type = chunk->type;

			auto& prev_chunk_info = prev_chunks_info_[kept_chunks];

			if (chunk_type != prev_chunk_info.type) break;

			// If we got media chunks, we compare its ID with the previous one.
			if (chunk_type == IMAGE || chunk_type == AUDIO) {
				if (chunk->id != prev_chunk_info.id) break;

				perfect_keep += chunk->n_tokens;
			}

			// If we got a text chunk, we compare its text_tokens with the previous ones, and partially keep the first differing chunk.
			else if (chunk_type == TEXT) {
				auto& tokens = chunk->text_tokens;

				while (
					last_keep < tokens.size() &&
					last_keep < prev_chunk_info.tokens.size() &&
					tokens[last_keep] == prev_chunk_info.tokens[last_keep]
					) ++last_keep;

				if (last_keep != tokens.size() || last_keep != prev_chunk_info.tokens.size()) break;

				perfect_keep += tokens.size();
				last_keep = 0;
			}

			else throw LlamaException("Invalid chunk type");

			kept_chunks += 1;
		}

		context_.KV_cleanup(perfect_keep + last_keep);

		if (last_keep != 0) {
			auto& last_chunk = chunks[kept_chunks];

			auto& tokens = last_chunk->text_tokens;

			try {
				context_.text_prefill({ tokens.begin() + last_keep, tokens.end() }, kept_chunks == chunks.size() - 1);
			}
			catch (const LlamaException& e) {
				prev_chunks_info_.clear(); // Reset to prevent any unpredictable state;
				throw LlamaException("Error prefilling cache: " + std::string(e.what()));
			}
		}

		try {
			context_.mtmd_prefill(chunks.subspan(kept_chunks + (last_keep != 0)));
		}
		catch (const LlamaException& e) {
			prev_chunks_info_.clear(); // Reset to prevent any unpredictable state;
			throw LlamaException("Error prefilling cache: " + std::string(e.what()));
		}

		// Update previous chunks info.
		prev_chunks_info_.resize(kept_chunks);
		prev_chunks_info_.reserve(chunks.size());

		if (last_keep != 0) {
			auto& chunk = chunks[kept_chunks];
			auto& tokens = chunk->text_tokens;
			auto& prev_tokens = prev_chunks_info_[kept_chunks].tokens;
			prev_tokens.insert(prev_tokens.end(), tokens.begin() + last_keep, tokens.end());
		}

		for (auto& chunk : chunks | std::views::drop(kept_chunks + (last_keep != 0))) {
			auto chunk_type = chunk->type;

			if (chunk_type == IMAGE || chunk_type == AUDIO) {
				prev_chunks_info_.emplace_back(ChunkInfo{ chunk_type, chunk->id, std::vector<llama_token>() });
			}
			else if (chunk_type == TEXT) {
				size_t n_tokens;
				auto& tokens = chunk->text_tokens;
				prev_chunks_info_.emplace_back(ChunkInfo{ chunk_type, std::string(), tokens });
			}
		}

		log_info(std::format("KV Cache used: {}", context_.get_used_memory()));
	}

	void KVScheduler::clear() {
		prev_tokens_.clear();
	}

}