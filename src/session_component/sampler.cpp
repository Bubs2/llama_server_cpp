#include "sampler.h"
#include "llama_log.h"
#include "llama_context.h"
#include "common.h"

#include <random>

namespace llama_server::internal {

	// ===================================================================
	// Sampler::SamplerDeleter
	// ===================================================================

	void Sampler::SamplerDeleter::operator()(llama_sampler* sampler) const { llama_sampler_free(sampler); }

	// ===================================================================
	// Sampler
	// ===================================================================

	Sampler::Sampler(const LlamaContext& context)
		: context_(context) {
		candidates_buffer_.resize(context_.get_n_vocab());
	}
	Sampler::~Sampler() = default;

	void Sampler::set(
		const GenConfig& gen_config,
		const Grammar& auto_grammar
	) {
		ptr_ = SamplerPtr(llama_sampler_chain_init(llama_sampler_chain_default_params()));

		if (!gen_config.grammar.value.empty()) {
			try { llama_sampler_chain_add(ptr_.get(), get_grammar_sampler(gen_config.grammar, context_.get_vocab())); }
			catch (const LlamaException& e) { log_warn(e.what()); }
		}
		else if (!auto_grammar.value.empty()) {
			try { llama_sampler_chain_add(ptr_.get(), get_grammar_sampler(auto_grammar, context_.get_vocab())); }
			catch (const LlamaException& e) { log_warn(e.what()); }
		}

		llama_sampler_chain_add(ptr_.get(), llama_sampler_init_temp(gen_config.temperature));
		if (gen_config.top_k != 0) {
			llama_sampler_chain_add(ptr_.get(), llama_sampler_init_top_k(gen_config.top_k));
		}
		if (gen_config.top_p > 0.0f && gen_config.top_p < 1.0f) {
			llama_sampler_chain_add(ptr_.get(), llama_sampler_init_top_p(gen_config.top_p, 10));
		}
		if (gen_config.penalty_last_n != 0) {
			llama_sampler_chain_add(ptr_.get(),
				llama_sampler_init_penalties(
					gen_config.penalty_last_n,
					gen_config.penalty_repeat,
					gen_config.penalty_freq,
					gen_config.penalty_present
				));
		}
		llama_sampler_chain_add(ptr_.get(), llama_sampler_init_dist(std::random_device()()));
	}

	llama_token Sampler::apply() {
		const float* logits = llama_get_logits_ith(context_.get_data(), -1);

		for (llama_token token_id = 0; token_id < context_.get_n_vocab(); token_id++) {
			candidates_buffer_[token_id] = llama_token_data{ token_id, logits[token_id], 0.0f };
		}

		llama_token_data_array data_array = {
			.data = candidates_buffer_.data(),
			.size = candidates_buffer_.size(),
			.selected = -1,
			.sorted = false
		};

		llama_sampler_apply(ptr_.get(), &data_array);

		if (data_array.selected < 0 || data_array.selected >= (int32_t)data_array.size) {
			throw ContextGenerateDirtyException(std::format("Sampler failed to select a token. Returned value: {}", data_array.selected));
		}

		llama_token result_token = data_array.data[data_array.selected].id;

		llama_sampler_accept(ptr_.get(), result_token);

		return result_token;
	}

	// Copy from sampling.cpp
	llama_sampler* Sampler::get_grammar_sampler(const Grammar& grammar, const llama_vocab* vocab) {
		struct llama_sampler* grmr;
		if (grammar.value.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
			grmr = llama_sampler_init_llg(vocab, "lark", grammar.value.c_str());
#else
			GGML_ABORT("llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
		}
		else {
			std::vector<std::string> trigger_patterns;
			std::vector<std::string> patterns_anywhere;
			std::vector<llama_token> trigger_tokens;
			for (const auto& trigger : grammar.triggers) {
				switch (trigger.type) {
				case GrammarTrigger::WORD:
				{
					const auto& word = trigger.value;
					patterns_anywhere.push_back(regex_escape(word));
					break;
				}
				case GrammarTrigger::PATTERN:
				{
					patterns_anywhere.push_back(trigger.value);
					break;
				}
				case GrammarTrigger::PATTERN_FULL:
				{
					trigger_patterns.push_back(trigger.value);
					break;
				}
				case GrammarTrigger::TOKEN:
				{
					const auto token = trigger.token;
					trigger_tokens.push_back(token);
					break;
				}
				default:
					GGML_ASSERT(false && "unknown trigger type");
				}
			}

			if (!patterns_anywhere.empty()) {
				trigger_patterns.push_back("^[\\s\\S]*?(" + string_join(patterns_anywhere, "|") + ")[\\s\\S]*");
			}

			std::vector<const char*> trigger_patterns_c;
			trigger_patterns_c.reserve(trigger_patterns.size());
			for (const auto& regex : trigger_patterns) {
				trigger_patterns_c.push_back(regex.c_str());
			}

			grmr = grammar.lazy
				? llama_sampler_init_grammar_lazy_patterns(vocab, grammar.value.c_str(), "root",
					trigger_patterns_c.data(), trigger_patterns_c.size(),
					trigger_tokens.data(), trigger_tokens.size())
				: llama_sampler_init_grammar(vocab, grammar.value.c_str(), "root");

			if (!grmr) {
				throw LlamaException("Failed to initialize grammar sampler");
			}

			return grmr;
		}
	}

}