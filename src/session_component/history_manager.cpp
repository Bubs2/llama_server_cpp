#include "history_manager.h"
#include "history_manager_internal.h"
#include "utils.h"
#include "llama_log.h"
#include "id_chunk.h"
#include "llama_model.h"
#include "templater.h"
#include "tokenizer.h"
#include "mtmd-helper.h"
#include "chat.h"

#include <unordered_set>
#include <string_view>
#include <format>
#include <numeric>
#include <re2/re2.h>

namespace llama_server {

	using namespace internal;
	using namespace history_manager_details;

	namespace history_manager_details {

		const re2::RE2 capture_path{ R"(<__path:(.*?)__>)" };

		std::vector<common_chat_tool> empty_tools;
		mtmd::bitmaps empty_bitmaps;

		class MediaHelper {
		public:
			MediaHelper(std::shared_ptr<LlamaModel> model)
				: model_(model) {
			}
			~MediaHelper() = default;

			std::string_view get_marker() const { return mtmd_default_marker(); }

			std::unique_ptr<mtmd::bitmaps> load_media(std::string_view path) {
				std::string path_str(path);

				std::unique_ptr<mtmd::bitmaps> result = std::make_unique<mtmd::bitmaps>();

				mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(model_->get_mtmd(), path_str.data()));
				if (!bmp.ptr) throw llama_server::LlamaException(std::format("Failed to load media from file: {}", path_str));
				result->entries.emplace_back(std::move(bmp));

				return result;
			}
		private:
			std::shared_ptr<LlamaModel> model_;
		};

		// ===================================================================
		// MessageManager
		// ===================================================================

		class MessageManager {
		public:
			MessageManager() = default;
			~MessageManager() = default;

			void add_message(common_chat_msg msg, bool is_head_prompt) {
				if (!is_head_prompt) {
					messages_.emplace_back(std::move(msg));
					return;
				}
				messages_.insert(messages_.begin() + head_prompt_size_, std::move(msg));
				head_prompt_size_++;
			}
			void add_tool(common_chat_tool tool) { tools_.emplace_back(std::move(tool)); }
			size_t get_message_size() const { return messages_.size(); }
			size_t get_head_prompt_size() const { return head_prompt_size_; }
			std::vector<common_chat_msg>& get_messages_ref() { return messages_; }
			std::vector<common_chat_tool>& get_tools_ref() { return tools_; }
		private:
			std::vector<common_chat_msg> messages_;
			std::vector<common_chat_tool> tools_;
			size_t head_prompt_size_ = 0;
		};

		// ===================================================================
		// ChunkManager
		// ===================================================================

		class ChunkManager {
		public:
			ChunkManager(
				std::shared_ptr<Templater> templater,
				std::shared_ptr<Tokenizer> tokenizer,
				std::unique_ptr<MediaHelper> media_helper
			) : templater_(templater), tokenizer_(tokenizer), media_helper_(std::move(media_helper)) {
			}
			~ChunkManager() = default;

			std::vector<IDChunkPtr> get_chunks(
				std::vector<common_chat_msg>& messages_ref,
				std::vector<common_chat_tool>& tools_ref,
				std::unique_ptr<common_chat_params>& chat_params_cache
			) {
				// We'll manually split the chunk, so add bos/eos here, instead of in tokenization.
				common_chat_templates_inputs inputs = {
					.add_bos = true,
					.add_eos = true
				};

				{
					auto message_guard = LoanGuard<std::vector<common_chat_msg>>(messages_ref, inputs.messages);
					auto tool_guard = LoanGuard<std::vector<common_chat_tool>>(tools_ref, inputs.tools);

					try { chat_params_cache = std::make_unique<common_chat_params>(templater_->apply_templates(inputs)); }
					catch (const LlamaException& e) { log_error(std::format("Failed to calc message tokens: {}", e.what())); }
				}

				std::vector<IDChunkPtr> result;
				auto add_text = [&result, this](std::string_view text) {
					auto text_tokens = tokenizer_->tokenize(text, false);
					result.emplace_back(std::make_shared<IDChunk>(std::move(text_tokens)));
				};
				auto add_media = [&result, this](std::string_view path) {
					auto it = image_chunk_cache_.find(path);

					if (it != image_chunk_cache_.end()) {
						result.emplace_back(image_start_chunk_);
						result.emplace_back(*it);
						result.emplace_back(image_end_chunk_);
					}

					else {
						std::unique_ptr<mtmd::bitmaps> bitmaps_ptr;
						try { bitmaps_ptr = media_helper_->load_media(path); }
						catch (const LlamaException& e) { log_error(std::format("Failed to load media from file: {}", e.what())); return; }

						auto chunks = tokenizer_->mtmd_tokenize(media_helper_->get_marker(), *bitmaps_ptr, false);
						auto chunk = mtmd_input_chunks_get(chunks.get(), 1);

						// Store chunk in cache
						auto [inserted_it, success] = image_chunk_cache_.emplace(std::make_shared<IDChunk>(std::string(path), chunk));
						image_start_chunk_ = std::make_shared<IDChunk>(std::string(), mtmd_input_chunks_get(chunks.get(), 0));
						image_end_chunk_ = std::make_shared<IDChunk>(std::string(), mtmd_input_chunks_get(chunks.get(), 2));

						// Emplace back the ptr
						result.emplace_back(image_start_chunk_);
						result.emplace_back(*inserted_it);
						result.emplace_back(image_end_chunk_);
					}
				};

				re2::StringPiece sp(chat_params_cache->prompt.data(), chat_params_cache->prompt.size());
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

			void unload_cache(std::string_view path) { image_chunk_cache_.erase(path); }
		private:
			std::shared_ptr<Templater> templater_;
			std::shared_ptr<Tokenizer> tokenizer_;
			std::unique_ptr<MediaHelper> media_helper_;

			std::unordered_set<IDChunkPtr, IDChunkPtrHash, IDChunkPtrEqual> image_chunk_cache_;
			IDChunkPtr image_start_chunk_ = nullptr;
			IDChunkPtr image_end_chunk_ = nullptr;
		};

	}

	// ===================================================================
	// HistoryManager
	// ===================================================================

	HistoryManager::HistoryManager(
		std::shared_ptr<LlamaModel> model,
		std::shared_ptr<Tokenizer> tokenizer,
		std::shared_ptr<Templater> templater,
		size_t context_size
	) : tokenizer_(tokenizer), templater_(templater), context_size_(context_size) {
		message_manager_ = std::make_unique<MessageManager>();

		std::unique_ptr<MediaHelper> media_helper = std::make_unique<MediaHelper>(model);
		chunk_manager_ = std::make_unique<ChunkManager>(templater, tokenizer, std::move(media_helper));
	}

	HistoryManager::~HistoryManager() = default;

	void HistoryManager::add_message(Message msg, bool is_head_prompt) {
		message_manager_->add_message(
			common_chat_msg{
				.role = std::move(msg.role),
				.content = std::move(msg.content),
			},
			is_head_prompt
			);
		if (!is_head_prompt) {
			size_t pos = message_manager_->get_message_size() - 1;
			estimate_msg_info(pos);
			total_token_estimate_ += msg_token_counts_[pos];
			return;
		}
		size_t pos = message_manager_->get_head_prompt_size() - 1;
		msg_token_counts_.insert(msg_token_counts_.begin() + pos, 0);
		estimate_msg_info(pos);
		total_token_estimate_ += msg_token_counts_[pos];
	}

	void HistoryManager::add_tool(Tool tool) {
		total_token_estimate_ -= tools_token_count_;
		message_manager_->add_tool(
			common_chat_tool{
				.name = std::move(tool.name),
				.description = std::move(tool.description),
				.parameters = std::move(tool.parameters)
			}
		);
		estimate_tools_info();
		total_token_estimate_ += tools_token_count_;
	}

	void HistoryManager::unload_cache(size_t spare) {
		if (total_token_estimate_ + spare <= context_size_) return;
		if (tools_token_count_ + spare > context_size_) throw LlamaException("Too much tools!");

		auto unload_images = [this](common_chat_msg& msg) {
			re2::StringPiece sp(msg.content.data(), msg.content.size());
			re2::StringPiece match[2]; // match[0] = full match, match[1] = capture group
			while (capture_path.Match(sp, 0, sp.size(), re2::RE2::UNANCHORED, match, 2)) {
				size_t prefix_len = match[0].data() - sp.data();
				chunk_manager_->unload_cache(match[1]);
				sp.remove_prefix(prefix_len + match[0].size());
			}
		};

		static const size_t margin = 128;
		total_token_estimate_ = std::reduce(
			msg_token_counts_.begin(),
			msg_token_counts_.begin() + message_manager_->get_head_prompt_size(),
			margin + spare + tools_token_count_
		);

		if (total_token_estimate_ > context_size_) {
			log_warn("Too much head prompts, unloading all normal prompts and tail head prompts.");

			auto& messages_ref = message_manager_->get_messages_ref();
			while (total_token_estimate_ > context_size_) {
				unload_images(messages_ref.back());

				total_token_estimate_ -= msg_token_counts_.back();
				msg_token_counts_.pop_back();
				messages_ref.pop_back();
			}

			return;
		}

		total_token_estimate_ = std::reduce(msg_token_counts_.begin() + message_manager_->get_head_prompt_size(), msg_token_counts_.end(), total_token_estimate_);

		if (total_token_estimate_ <= context_size_) return;

		auto& messages_ref = message_manager_->get_messages_ref();
		size_t head_prompt_size = message_manager_->get_head_prompt_size();
		while (total_token_estimate_ > context_size_) {
			unload_images(messages_ref[head_prompt_size]);

			total_token_estimate_ -= msg_token_counts_[head_prompt_size];
			msg_token_counts_.erase(msg_token_counts_.begin() + head_prompt_size);
			messages_ref.erase(messages_ref.begin() + head_prompt_size);
		}
	}

	void HistoryManager::estimate_msg_info(size_t index) {
		auto& messages_ref = message_manager_->get_messages_ref();
		if (index >= messages_ref.size()) throw std::out_of_range("Index out of range");
		msg_token_counts_.resize(messages_ref.size());

		auto loan = [](common_chat_msg& msg, std::vector<common_chat_msg>& vec) { vec.emplace_back(std::move(msg)); };
		auto ret = [](common_chat_msg& msg, std::vector<common_chat_msg>& vec) { msg = std::move(vec[0]); };

		{
			std::vector<common_chat_msg> messages;

			auto guard = LoanGuard<common_chat_msg, std::vector<common_chat_msg>>(messages_ref[index], messages, loan, ret);

			std::vector<IDChunkPtr> chunks = chunk_manager_->get_chunks(messages, empty_tools, chat_params_cache_);
			msg_token_counts_[index] = std::transform_reduce(
				chunks.begin(), chunks.end(),
				size_t{ 0 }, std::plus<>(),
				[](const IDChunkPtr item) { return item->get_n_tokens(); }
			);
		}
	}

	void HistoryManager::estimate_tools_info() {
		common_chat_templates_inputs inputs = {
			.messages = { {.role = "system", .content = "Placeholder to prevent tokens exceeding." } }
		};

		{
			auto tools_guard = internal::LoanGuard<std::vector<common_chat_tool>>(message_manager_->get_tools_ref(), inputs.tools);

			try { chat_params_cache_ = std::make_unique<common_chat_params>(templater_->apply_templates(inputs)); }
			catch (const LlamaException& e) { log_error(std::format("Failed to calc message tokens: {}", e.what())); }
		}

		std::vector<llama_token> tokens = tokenizer_->tokenize(chat_params_cache_->prompt);
		tools_token_count_ = tokens.size();
	}

	common_chat_params& ChatParamsAccessor::get_params(HistoryManager& hm) {
		return *hm.chat_params_cache_;
	}

	std::vector<IDChunkPtr> ChunksAccessor::get_chunks(HistoryManager& hm, size_t max_tokens) {
		hm.unload_cache(max_tokens);

		return hm.chunk_manager_->get_chunks(
			hm.message_manager_->get_messages_ref(),
			hm.message_manager_->get_tools_ref(),
			hm.chat_params_cache_
		);
	}

}