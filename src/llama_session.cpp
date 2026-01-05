#include "llama_session.h"
#include "llama_log.h"
#include "llama_strategy.h"
#include "id_chunk.h"
#include "llama_model.h"
#include "llama_context.h"
#include "input_encoder.h"
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

		context_ = std::make_unique<LlamaContext>(llm_context_params, model);

		tokenizer_ = std::make_unique<Tokenizer>(context_->get_model());
		templater_ = std::make_unique<Templater>(context_->get_model());

		input_encoder_ = std::make_unique<InputEncoder>(*context_, *templater_, *tokenizer_, TokenEstimateStrategy(16));
		kv_scheduler_ = std::make_unique<KVScheduler>(*context_);

		sampler_ = std::make_unique<Sampler>(*context_);
		streamer_ = std::make_unique<Streamer>();
	}

	LlamaSession::~LlamaSession() { return; }

	LlamaSession::LlamaSession(LlamaSession&&) noexcept = default;
	LlamaSession& LlamaSession::operator=(LlamaSession&&) noexcept = default;

	void LlamaSession::generate(
		std::vector<Message> head_msgs,
		std::vector<Message> tail_msgs,
		std::vector<Tool> tools,
		const GenConfig& gen_config
	) {
		// Pre-checks
		size_t max_tokens = gen_config.max_tokens;

		if (gen_config.max_tokens == 0) {
			log_warn("Max tokens is set to 0, nothing to generate.");
			return;
		}
		if (gen_config.max_tokens > context_->get_n_ctx()) {
			log_warn("Max tokens is greater than context size, setting to context size. Note: all memory will be pruned.");
            max_tokens = context_->get_n_ctx();
		}

		// Translate Message & Tool wrapper
		std::vector<common_chat_msg> llama_head_msgs;
		for (auto& msg : head_msgs) {
			llama_head_msgs.emplace_back(common_chat_msg{
				.role = std::move(msg.role),
				.content = std::move(msg.content)
			});
		}
		std::vector<common_chat_msg> llama_tail_msgs;
		for (auto& msg : tail_msgs) {
			llama_tail_msgs.emplace_back(common_chat_msg{
				.role = std::move(msg.role),
				.content = std::move(msg.content)
			});
		}
		std::vector<common_chat_tool> llama_tools;
		for (auto& tool : tools) {
			llama_tools.emplace_back(common_chat_tool{
				.name = std::move(tool.name),
				.description = std::move(tool.description),
				.parameters = std::move(tool.parameters)
			});
		}

		// Prune and tokenize
		std::vector<IDChunksPtr> chunks;
		try { chunks = (*input_encoder_)(std::move(llama_head_msgs), std::move(llama_tail_msgs), std::move(llama_tools), max_tokens); }
		catch (const LlamaException& e) {
			log_error(e.what());
			return;
		}

		// Prefill
		try { kv_scheduler_->prefill_mtmd_cache(chunks); }
		catch (const LlamaException& e) {
			log_error(e.what());
			return;
		}

		common_chat_params chat_params = input_encoder_->get_chat_params_cache();
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
		while (n_generated < max_tokens) {
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

	void LlamaSession::set_token_estimate_strategy(TokenEstimateStrategy&& strategy) {
		input_encoder_->set_token_estimate_strategy(std::move(strategy));
	}

}