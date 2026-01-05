#include "tokenizer.h"
#include "llama_model.h"

#include <format>

namespace llama_server::internal {

	Tokenizer::Tokenizer(const LlamaModel& model)
		: model_(model) { }

	Tokenizer::~Tokenizer() { return; }

	std::vector<llama_token> Tokenizer::text_tokenize(
		std::string_view text,
		bool add_special,
		bool parse_special
	) const {
		int32_t estimated_size = text.size() + (add_special ? 2 : 0);
		const int32_t try_cap = 2;

		std::vector<llama_token> tokens;

		int32_t n_tokens;
		for (int32_t i = 0; i < try_cap; i++) {
			tokens.resize(estimated_size);

			n_tokens = llama_tokenize(
				model_.get_vocab(),
				text.data(),
				text.length(),
				tokens.data(),
				tokens.size(),
				add_special,
				parse_special
			);

			if (n_tokens >= 0) break;
			if (n_tokens == INT32_MIN) throw LlamaException("Tokenization overflow, exceeds int32_t limit.");
			estimated_size = -n_tokens;
		}

		if (n_tokens < 0) throw LlamaException(std::format("Tokenization overflow, final estimated_size = {}, result_size = {}", estimated_size, -n_tokens));

		tokens.resize(n_tokens);

		return tokens;
	}

	std::string Tokenizer::detokenize(
		llama_token token,
		bool add_special
	) const {
		std::string buffer(256, '\0');

		const int32_t n_chars = llama_token_to_piece(
			model_.get_vocab(),
			token,
			buffer.data(),
			buffer.size(),
			0,
			add_special
		);

		if (n_chars < 0) throw LlamaException("Failed to detokenize token.");

		buffer.resize(n_chars);

		return buffer;
	}

	mtmd::input_chunks_ptr Tokenizer::mtmd_tokenize(
		std::string_view text,
		mtmd::bitmaps& bitmaps,
		bool add_special,
		bool parse_special
	) const {
		std::string text_str(text);

		mtmd_input_text mtmd_text{
			.text = text_str.data(),
			.add_special = add_special,
			.parse_special = parse_special
		};

		auto bitmaps_c_ptr = bitmaps.c_ptr();

		mtmd::input_chunks_ptr chunks_ptr(mtmd_input_chunks_init());

		int32_t res = ::mtmd_tokenize(
			model_.get_mtmd(),
			chunks_ptr.get(), // output
			&mtmd_text, // text
			bitmaps_c_ptr.data(),
			bitmaps_c_ptr.size()
		);

		switch (res) {
		case 0:
			break;
		case 1:
			throw LlamaException("Number of bitmaps does not match number of markers.");
		case 2:
			throw LlamaException("Image preprocessing error.");
		default:
			throw LlamaException("Unknown error.");
		}

		return chunks_ptr;
	}

}