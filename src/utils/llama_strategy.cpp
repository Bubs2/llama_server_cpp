#include "llama_strategy.h"

namespace llama_server {

	// ===================================================================
	// TokenEstimateStrategy
	// ===================================================================

	struct TokenEstimateStrategy::Impl {
		Impl(EstimateCallback&& estimate_cb) : estimate_cb(std::move(estimate_cb)) {};
		~Impl() = default;

		const EstimateCallback estimate_cb;
	};

	TokenEstimateStrategy::TokenEstimateStrategy(size_t margin)
		: impl_(std::make_unique<Impl>([](const TokenizeCallback& tokenize_cb, std::string_view str) { return tokenize_cb(str); })), margin_(margin) {};

	TokenEstimateStrategy::TokenEstimateStrategy(EstimateCallback estimate_cb, size_t margin)
		: impl_(std::make_unique<Impl>(std::move(estimate_cb))), margin_(margin) {}

	TokenEstimateStrategy::~TokenEstimateStrategy() = default;

	TokenEstimateStrategy::TokenEstimateStrategy(TokenEstimateStrategy&&) noexcept = default;
    TokenEstimateStrategy& TokenEstimateStrategy::operator=(TokenEstimateStrategy&&) noexcept = default;

	size_t TokenEstimateStrategy::estimate(const TokenizeCallback& tokenize_cb, std::string_view str) const { return impl_->estimate_cb(tokenize_cb, str); }
}