#pragma once

#include "chat.h"

#include <memory>

namespace llama_server::internal {

	class LlamaModel;

	class Templater {
	public:
		Templater(const LlamaModel& model);
		~Templater();

		common_chat_params operator()(const common_chat_templates_inputs& inputs) const;
	private:
		const common_chat_templates_ptr tmpl_;
	};

}