#pragma once

#include "mtmd.h"

#include <memory>
#include <vector>
#include <string>
#include <string_view>

namespace llama_server::internal {

	using llama_token = int32_t;

	enum ChunkType {
		TEXT,
		IMAGE,
		AUDIO,
	};

	struct IDChunk {
		struct ChunkDeleter {
			void operator()(mtmd_input_chunk* ptr) { mtmd_input_chunk_free(ptr); }
		};

		ChunkType type = (ChunkType)100;

		std::string id;
		std::unique_ptr<mtmd_input_chunk, ChunkDeleter> chunk = nullptr;

		std::vector<llama_token> text_tokens;

		IDChunk() = default;
		explicit IDChunk(std::string id, const mtmd_input_chunk* chunk)
			: id(id)
		{
			switch (mtmd_input_chunk_get_type(this->chunk.get())) {
			case MTMD_INPUT_CHUNK_TYPE_TEXT: {
				type = TEXT;
				size_t n_tokens;
				const llama_token* ptr = mtmd_input_chunk_get_tokens_text(this->chunk.get(), &n_tokens);
				text_tokens.assign(ptr, ptr + n_tokens);
				break;
			}
			case MTMD_INPUT_CHUNK_TYPE_IMAGE:
                type = IMAGE;
				this->chunk = std::unique_ptr<mtmd_input_chunk, ChunkDeleter>(mtmd_input_chunk_copy(chunk));
				break;
			case MTMD_INPUT_CHUNK_TYPE_AUDIO:
				type = AUDIO;
				this->chunk = std::unique_ptr<mtmd_input_chunk, ChunkDeleter>(mtmd_input_chunk_copy(chunk));
				break;
			}
		}
		explicit IDChunk(std::vector<llama_token> tokens)
			: text_tokens(std::move(tokens))
		{ type = TEXT; }

		IDChunk(IDChunk&&) = default;
		IDChunk& operator=(IDChunk&&) = default;
		IDChunk(const IDChunk&) = delete;
		IDChunk& operator=(const IDChunk&) = delete;

		const mtmd_input_chunk* get_data() const { return chunk.get(); }
		size_t get_n_tokens() const {
			if (type == TEXT) return text_tokens.size();
			return mtmd_input_chunk_get_n_tokens(chunk.get());
		}
	};

	using IDChunkPtr = std::shared_ptr<IDChunk>;

	struct IDChunkPtrHash {
		using is_transparent = void;
		size_t operator()(const IDChunkPtr& ptr) const {
			if (!ptr) return 0;
			return std::hash<std::string_view>{}(ptr->id);
		}
		size_t operator()(std::string_view id) const {
			return std::hash<std::string_view>{}(id);
		}
	};

	struct IDChunkPtrEqual {
		using is_transparent = void;
		bool operator()(const IDChunkPtr& lhs, const IDChunkPtr& rhs) const {
			if (lhs == rhs) return true;
			if (!lhs || !rhs) return false;
			return lhs->id == rhs->id;
		}
		bool operator()(const IDChunkPtr& lhs, std::string_view rhs) const {
			return lhs && lhs->id == rhs;
		}
		bool operator()(std::string_view lhs, const IDChunkPtr& rhs) const {
			return rhs && lhs == rhs->id;
		}
	};

}