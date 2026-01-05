#include "templater.h"
#include "llama_model.h"

namespace llama_server::internal {

	Templater::Templater(const LlamaModel& model)
		: tmpl_(common_chat_templates_init(model.get_data(), "")) { }

	Templater::~Templater() {}

	common_chat_params Templater::operator()(const common_chat_templates_inputs& inputs) const {
		return common_chat_templates_apply(tmpl_.get(), inputs);
	}

}