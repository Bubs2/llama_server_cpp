#include "input_encoder.h"
#include "utils.h"
#include "llama_log.h"
#include "llama_context.h"
#include "llama_model.h"
#include "templater.h"
#include "tokenizer.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <re2/re2.h>
#include <string>
#include <ranges>

namespace llama_server::internal {

	using namespace input_encoder_detail;

	namespace input_encoder_detail {

		const re2::RE2 capture_path{ R"(<__path:(.*?)__>)" };

		common_chat_msg guard_prompt = { .role = "system", .content = "This is a guard to prevent tokens exceeding the context length." };

		std::vector<common_chat_tool> empty_tools;
		mtmd::bitmaps empty_bitmaps;

		class MediaHelper {
		public:
			MediaHelper(const LlamaModel& model)
				: model_(model) {
			}
			~MediaHelper() = default;

			std::unique_ptr<mtmd::bitmaps> load_media(std::string_view path) {
				std::string path_str(path);

				std::unique_ptr<mtmd::bitmaps> result = std::make_unique<mtmd::bitmaps>();

				mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(model_.get_mtmd(), path_str.data()));
				if (!bmp.ptr) throw llama_server::LlamaException(std::format("Failed to load media from file: {}", path_str));
				result->entries.emplace_back(std::move(bmp));

				return result;
			}
		private:
			const LlamaModel& model_;
		};

	}

	InputEncoder::InputEncoder(
		const LlamaContext& context,
		const Templater& templater,
		const Tokenizer& tokenizer,
		TokenEstimateStrategy&& token_estimate_strategy
	) : context_(context), templater_(templater), tokenizer_(tokenizer), token_estimate_strategy_(std::move(token_estimate_strategy)) {
		media_helper_ = std::make_unique<MediaHelper>(context_.get_model());
	}
	
	InputEncoder::~InputEncoder() = default;

	std::vector<IDChunksPtr> InputEncoder::operator()(
		std::vector<common_chat_msg>&& head_msgs,
		std::vector<common_chat_msg>&& tail_msgs,
		std::vector<common_chat_tool>&& tools,
		size_t max_tokens
	) {
		if (image_chunks_cache_.size() > 16) image_chunks_cache_.clear(); // Temporary simple cache eviction

		common_chat_templates_inputs input = prune_with_precache(std::move(head_msgs), std::move(tail_msgs), std::move(tools), max_tokens);
		chat_params_cache_ = templater_(input);
		used_messages_cache_ = input.messages.size();

		std::vector<IDChunksPtr> result;
		auto add_text = [&result, this](std::string_view text) {
			auto text_tokens = tokenizer_.text_tokenize(text, false);
			result.emplace_back(std::make_shared<IDChunks>(std::move(text_tokens)));
		};
		auto add_media = [&result, this](std::string_view path) {
			auto it = image_chunks_cache_.find(path);

			if (it == image_chunks_cache_.end()) {
				std::unique_ptr<mtmd::bitmaps> bitmaps_ptr = media_helper_->load_media(path);

				auto chunks = tokenizer_.mtmd_tokenize(mtmd_default_marker(), *bitmaps_ptr, false);
				// Store chunk in cache
				it = image_chunks_cache_.emplace(std::make_shared<IDChunks>(std::string(path), std::move(chunks))).first;
			}

			result.emplace_back(*it);
		};

		re2::StringPiece sp(chat_params_cache_.prompt.data(), chat_params_cache_.prompt.size());
		re2::StringPiece match[2]; // match[0] = full match, match[1] = capture group
		while (capture_path.Match(sp, 0, sp.size(), re2::RE2::UNANCHORED, match, 2)) {
			size_t prefix_len = match[0].data() - sp.data();

			if (prefix_len) add_text(std::string_view(sp.data(), prefix_len));
			add_media(std::string_view(match[1].data(), match[1].size()));

			sp.remove_prefix(prefix_len + match[0].size());
		}

		if (!sp.empty()) add_text(std::string_view(sp.data(), sp.size()));

		return result;
	}

	size_t InputEncoder::get_used_messages_cache() { return used_messages_cache_; }
	common_chat_params InputEncoder::get_chat_params_cache() { return std::move(chat_params_cache_); }

	size_t InputEncoder::estimate_text_tokens(std::string_view str) { return tokenizer_.text_tokenize(str).size(); }

	size_t InputEncoder::estimate_mtmd_tokens(std::string_view str) {
		size_t n_tokens = 0;

		re2::StringPiece sp(str);
		re2::StringPiece match[2]; // match[0] = full match, match[1] = capture group
		while (capture_path.Match(sp, 0, sp.size(), re2::RE2::UNANCHORED, match, 2)) {
			size_t prefix_len = match[0].data() - sp.data();

			if (prefix_len) n_tokens += tokenizer_.text_tokenize(std::string_view(sp.data(), prefix_len)).size();

			std::string_view path(match[1].data(), match[1].size());
			auto it = image_chunks_cache_.find(path);
			if (it == image_chunks_cache_.end()) {
				std::unique_ptr<mtmd::bitmaps> bitmaps_ptr = media_helper_->load_media(path);

				auto chunks = tokenizer_.mtmd_tokenize(mtmd_default_marker(), *bitmaps_ptr, false);
				// Store chunk in cache
				it = image_chunks_cache_.emplace(std::make_shared<IDChunks>(std::string(path), std::move(chunks))).first;
			}

			n_tokens += (*it)->n_tokens;

			sp.remove_prefix(prefix_len + match[0].size());
		}

		if (!sp.empty()) n_tokens += tokenizer_.text_tokenize(std::string_view(sp.data(), sp.size())).size();

		return n_tokens;
	}

	common_chat_templates_inputs InputEncoder::prune_with_precache(
		std::vector<common_chat_msg>&& head_msgs,
		std::vector<common_chat_msg>&& tail_msgs,
		std::vector<common_chat_tool>&& tools,
		size_t max_tokens
	) {
		const TokenizeCallback tokenize_callback = context_.is_mtmd()
			? TokenizeCallback([this](std::string_view s) { return estimate_mtmd_tokens(s); })
			: TokenizeCallback([this](std::string_view s) { return estimate_text_tokens(s); });

		common_chat_templates_inputs result = { .add_bos = true, .add_eos = true };
		common_chat_templates_inputs temporary_inputs = { .messages = std::vector<common_chat_msg>(1), .add_generation_prompt = false };
		size_t n_cap = context_.get_n_ctx() - max_tokens;

		size_t n_tokens = token_estimate_strategy_.margin_;
		{
			
			LoanGuard messages_guard(guard_prompt, temporary_inputs.messages[0]);
			LoanGuard tools_guard(tools, temporary_inputs.tools);

			common_chat_params params = templater_(temporary_inputs);
			n_tokens += token_estimate_strategy_.estimate(tokenize_callback, params.prompt);
		}

		if (n_tokens > n_cap) { throw llama_server::LlamaException("Too many tools tokens"); }
		result.tools = std::move(tools);

		for (auto& head_msg : head_msgs) {
			{
				LoanGuard message_guard(head_msg, temporary_inputs.messages[0]);

				common_chat_params params = templater_(temporary_inputs);
				n_tokens += token_estimate_strategy_.estimate(tokenize_callback, params.prompt);
			}
			if (n_tokens > n_cap) {
				log_warn("Too many head messages tokens, dropping messages.");
				return result;
			}

			result.messages.emplace_back(std::move(head_msg));
		}

		std::vector<common_chat_msg> reverse_queue;
		for (auto& tail_msg : tail_msgs | std::views::reverse) {
			{
				LoanGuard message_guard(tail_msg, temporary_inputs.messages[0]);

				common_chat_params params = templater_(temporary_inputs);
				n_tokens += token_estimate_strategy_.estimate(tokenize_callback, params.prompt);
			}
			if (n_tokens > n_cap) break;

			reverse_queue.emplace_back(std::move(tail_msg));
		}

        for (auto& tail_msg : reverse_queue | std::views::reverse) result.messages.emplace_back(std::move(tail_msg));

		return result;
	}
}