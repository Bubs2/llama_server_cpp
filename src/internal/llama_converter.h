#pragma once

#include "llama_configs.h"
#include "common.h"
#include "chat.h"

namespace llama_server::internal {

	struct GrammarConverter {
		static inline Grammar normalize(common_chat_params& src) {
			Grammar dst = { .value = std::move(src.grammar), .lazy = src.grammar_lazy, };
			for (auto& trigger : src.grammar_triggers) {
				GrammarTrigger t = { .value = std::move(trigger.value), .token = trigger.token };
				switch (trigger.type) {
				case COMMON_GRAMMAR_TRIGGER_TYPE_WORD: t.type = GrammarTrigger::WORD; break;
				case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN: t.type = GrammarTrigger::TOKEN; break;
				case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN: t.type = GrammarTrigger::PATTERN; break;
				case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL: t.type = GrammarTrigger::PATTERN_FULL; break;
				}
				dst.triggers.emplace_back(std::move(t));
			}
			return dst;
		}
	};

}