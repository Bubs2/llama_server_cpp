#pragma once

#include <string>
#include <string_view>
#include <functional>

namespace llama_server::internal {

	class Streamer {
	public:
		Streamer();
		~Streamer() = default;

		using StreamCallback = std::function<bool(std::string&&)>;
		bool process(std::string& buffer, StreamCallback callback);
	private:
		bool validate_utf8_end(std::string_view buffer);
	};

}
