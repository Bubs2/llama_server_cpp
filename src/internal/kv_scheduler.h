#pragma once

#include "llama_exception.h"
#include "id_chunk.h"
#include "llama.h"
#include "mtmd.h"

#include <memory>
#include <vector>
#include <span>

namespace llama_server::internal {

    class LlamaContext;

    class KVScheduler {
    public:
        KVScheduler(std::shared_ptr<LlamaContext> context);
        ~KVScheduler() = default;

        [[deprecated("text cache && mtmd cache uses diffirent inner buffer, DO NOT intermix them!")]]
        void prefill_text_cache(std::vector<llama_token> tokens);
        void prefill_mtmd_cache(std::span<IDChunkPtr const> chunks);
        void clear();
    private:
        struct ChunkInfo {
            ChunkType type = (ChunkType)99;
            std::string id;
            std::vector<llama_token> tokens;
        };

        std::shared_ptr<LlamaContext> context_;

        // For text generation.
        std::vector<llama_token> prev_tokens_;

        std::vector<ChunkInfo> prev_chunks_info_;
    };

}