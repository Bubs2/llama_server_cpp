#pragma once

#include "mtmd.h"

#include <memory>
#include <vector>
#include <string>
#include <string_view>
#include <ranges>

namespace llama_server::internal {

	using llama_token = int32_t;

	enum ChunksType {
		TEXT,
		IMAGE,
		AUDIO,
	};

	struct IDChunks {
		ChunksType type = (ChunksType)100;

		std::string id;
		mtmd::input_chunks_ptr chunks;
		std::vector<llama_token> text_tokens;
		size_t n_tokens = 0;

		explicit IDChunks(std::string id, mtmd::input_chunks_ptr chunks)
			: id(id), chunks(std::move(chunks))
		{
			size_t n_chunks = mtmd_input_chunks_size(this->chunks.get());
			for (auto idx : std::views::iota((size_t)0, n_chunks)) {
				n_tokens += mtmd_input_chunk_get_n_tokens(mtmd_input_chunks_get(this->chunks.get(), idx));
			}

			for (auto idx : std::views::iota((size_t)0, n_chunks)) {
				switch (mtmd_input_chunk_get_type(mtmd_input_chunks_get(this->chunks.get(), idx))) {
				case MTMD_INPUT_CHUNK_TYPE_IMAGE: type = IMAGE; return;
				case MTMD_INPUT_CHUNK_TYPE_AUDIO: type = AUDIO; return;
				}
			}

			for (auto idx : std::views::iota((size_t)0, n_chunks)) {
				size_t n_tokens;
				const llama_token* tokens = mtmd_input_chunk_get_tokens_text(mtmd_input_chunks_get(this->chunks.get(), idx), &n_tokens);
				text_tokens.insert(text_tokens.end(), tokens, tokens + n_tokens);
			}
			type = TEXT;
		}
		explicit IDChunks(std::vector<llama_token> tokens)
			: text_tokens(std::move(tokens))
		{ type = TEXT; n_tokens = text_tokens.size(); }

		IDChunks(IDChunks&&) = default;
		IDChunks& operator=(IDChunks&&) = default;
		IDChunks(const IDChunks&) = delete;
		IDChunks& operator=(const IDChunks&) = delete;
	};

	using IDChunksPtr = std::shared_ptr<IDChunks>;

	struct IDChunksPtrHash {
		using is_transparent = void;
		size_t operator()(const IDChunksPtr& ptr) const {
			if (!ptr) return 0;
			return std::hash<std::string_view>{}(ptr->id);
		}
		size_t operator()(std::string_view id) const {
			return std::hash<std::string_view>{}(id);
		}
	};

	struct IDChunksPtrEqual {
		using is_transparent = void;
		bool operator()(const IDChunksPtr& lhs, const IDChunksPtr& rhs) const {
			if (lhs == rhs) return true;
			if (!lhs || !rhs) return false;
			return lhs->id == rhs->id;
		}
		bool operator()(const IDChunksPtr& lhs, std::string_view rhs) const {
			return lhs && lhs->id == rhs;
		}
		bool operator()(std::string_view lhs, const IDChunksPtr& rhs) const {
			return rhs && lhs == rhs->id;
		}
	};

}