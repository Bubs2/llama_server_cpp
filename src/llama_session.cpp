#include "llama_session.h"
#include "llama_model.h"
#include "llama_context.h"
#include "llama_log.h"
#include "history_manager_internal.h"
#include "kv_scheduler.h"
#include "tokenizer.h"
#include "templater.h"
#include "sampler.h"
#include "streamer.h"
#include "llama.h"

#include <format>

namespace llama_server {

	using namespace internal;

	LlamaSession::LlamaSession(
		ContextConfig context_config,
		std::shared_ptr<LlamaModel> model
	) {
		llama_context_params llm_context_params = llama_context_default_params();
		llm_context_params.n_ctx = context_config.n_ctx;
		llm_context_params.n_batch = context_config.n_batch;
		llm_context_params.n_ubatch = context_config.n_ubatch;

		context_ = std::make_shared<LlamaContext>(llm_context_params, model);
		kv_scheduler_ = std::make_unique<KVScheduler>(context_);
		sampler_ = std::make_unique<Sampler>(context_);

		tokenizer_ = std::make_shared<Tokenizer>(context_->get_model());
		templater_ = std::make_shared<Templater>(context_->get_model());

		history_manager_ = std::make_unique<HistoryManager>(context_->get_model(), tokenizer_, templater_, context_->get_n_ctx());

		streamer_ = std::make_unique<Streamer>();
	}

	LlamaSession::~LlamaSession() { return; }

	LlamaSession::LlamaSession(LlamaSession&&) noexcept = default;
	LlamaSession& LlamaSession::operator=(LlamaSession&&) noexcept = default;

	void LlamaSession::generate(const GenConfig& gen_config) {
		// Pre-checks
		if (gen_config.max_tokens == 0) {
			log_warn("Max tokens is set to 0, nothing to generate.");
			return;
		}

		// Prefill
		try { kv_scheduler_->prefill_mtmd_cache(ChunksAccessor::get_chunks(*history_manager_, gen_config.max_tokens)); }
		catch (const LlamaException& e) {
			log_error(e.what());
			return;
		}

		common_chat_params chat_params = ChatParamsAccessor::get_params(*history_manager_);
		log_info(std::format("\n{}", chat_params.prompt));

		sampler_->set(gen_config, chat_params);

		llama_token next_token = sampler_->apply();
		if (llama_vocab_is_eog(context_->get_vocab(), next_token)) {
			log_info("End of generation");
			return;
		}

		std::string buffer = tokenizer_->detokenize(next_token);
		if (!streamer_->process(buffer, gen_config.output_callback)) return;

		// Generation loop
		size_t n_generated = 1;
		while (n_generated < gen_config.max_tokens) {
			try {
				context_->step(next_token);
			}
			catch (const LlamaException& e) {
				log_error(e.what());
				return;
			}

			if (llama_vocab_is_eog(context_->get_vocab(), next_token)) {
				log_info("End of generation");
				break;
			}

			next_token = sampler_->apply();
			buffer += tokenizer_->detokenize(next_token);
			if (!streamer_->process(buffer, gen_config.output_callback)) break;

			n_generated++;
		}

		return;
	}

	HistoryManager& LlamaSession::access_history_manager() {
		return *history_manager_;
	}

}