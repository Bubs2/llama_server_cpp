#pragma once

#include "llama_exception.h"
#include "id_chunk.h"
#include "llama.h"
#include "mtmd.h"

#include <memory>
#include <vector>
#include <span>

struct common_chat_msg;
struct common_chat_tool;

namespace llama_server::internal {

    class LlamaContext;
    class Templater;
    class Tokenizer;

    class KVScheduler {
    public:
        KVScheduler(
            std::shared_ptr<LlamaContext> context,
            std::shared_ptr<Tokenizer> tokenizer,
            std::shared_ptr<Templater> templater
        );
        ~KVScheduler() = default;

        void prefill_cache(
            std::vector<common_chat_msg> msgs,
            std::vector<common_chat_tool> tools,
            size_t max_tokens
        );
        void clear();
    private:
        struct ChunkInfo {
            ChunkType type = (ChunkType)99;
            std::string id;
            std::vector<llama_token> tokens;
        };

        std::shared_ptr<LlamaContext> context_;
        std::shared_ptr<Tokenizer> tokenizer_;
        std::shared_ptr<Templater> templater_;

        std::vector<ChunkInfo> prev_chunks_info_;
    };

}