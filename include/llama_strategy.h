#pragma once

#include <memory>
#include <functional>
#include <string_view>

namespace llama_server{

	namespace internal {
		class InputEncoder;
	}

	using TokenizeCallback = std::function<size_t(std::string_view str)>;
	using EstimateCallback = std::function<size_t(const TokenizeCallback& tokenize, std::string_view str)>;
	class TokenEstimateStrategy {
	public:
		TokenEstimateStrategy(size_t margin = 0);
		explicit TokenEstimateStrategy(EstimateCallback estimate, size_t margin = 0);
		~TokenEstimateStrategy();

		TokenEstimateStrategy(TokenEstimateStrategy&&) noexcept;
		TokenEstimateStrategy& operator=(TokenEstimateStrategy&&) noexcept;
		TokenEstimateStrategy(const TokenEstimateStrategy&) = delete;
		TokenEstimateStrategy& operator=(const TokenEstimateStrategy&) = delete;
	private:
		struct Impl;
		std::unique_ptr<Impl> impl_;

		size_t margin_;
		size_t estimate(const TokenizeCallback& tokenize_cb, std::string_view str) const;

		friend class internal::InputEncoder;
	};

}