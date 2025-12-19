#include "streamer.h"

#include <ranges>

namespace llama_server::internal {

	Streamer::Streamer() {}

	bool Streamer::process(
		std::string& buffer,
		StreamCallback callback
	) {
		if (validate_utf8_end(buffer)) {
			std::string result = std::move(buffer);
			buffer.clear();
			return callback(std::move(result));
		}
		return true;
	}

	bool Streamer::validate_utf8_end(std::string_view buffer) {
		if (buffer.size() == 0 || (buffer.back() & 0x80) == 0) return true;

		auto tail_view = buffer
			| std::views::reverse
			| std::views::take(4)
			| std::views::enumerate;

		for (auto [suf_len, byte_char] : tail_view) {
			if ((byte_char & 0xC0) == 0x80) continue;
			if ((byte_char & 0xE0) == 0xC0) return suf_len == 1;
			if ((byte_char & 0xF0) == 0xE0) return suf_len == 2;
			if ((byte_char & 0xF8) == 0xF0) return suf_len == 3;
			return false;
		}

		return false;
	}

}