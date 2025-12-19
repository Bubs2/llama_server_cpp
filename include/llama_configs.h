#pragma once

#include <string>
#include <functional>

namespace llama_server {

	struct ModelConfig {
		std::string model_path;
		int32_t n_gpu_layers = 32;		// number of layers to store in VRAM
		bool	use_mmap = true;		// use mmap if possible
		bool	use_mlock = false;		// force system to keep model in RAM

		std::string mtmd_path;			// path to multimodel
		int image_min_tokens = -1;		// minimum number of text_tokens for image input (default: read from metadata)
		int image_max_tokens = -1;		// maximum number of text_tokens for image input (default: read from metadata)
	};

	struct ContextConfig {
		uint32_t n_ctx = 8192;			// text context, 0 = from model
		uint32_t n_batch = 1024;		// logical maximum batch size that can be submitted to llama_decode
		uint32_t n_ubatch = 512;		// physical maximum batch size
	};

	using OutputCallback = std::function<bool(std::string&&)>;
	struct GenConfig {
		uint32_t	max_tokens = 1024;
		bool		enable_thinking = true;

		float		temperature = 1.0f;
		float		top_p = 0.0f;
		int32_t		top_k = 0;

		int32_t		penalty_last_n = 0;
		float		penalty_repeat = 1.0f;
		float		penalty_freq = 0.0f;
		float		penalty_present = 0.0f;

		bool		add_generation_prompt = true;
		OutputCallback output_callback = nullptr;
	};

	using ToolCallback = std::function<std::string(std::string_view json_str)>;

}