# llama.cpp_wrapper

A lightweight C++23 inference engine built on top of [llama.cpp](https://github.com/ggerganov/llama.cpp). This library works as a wrapper and a basic KV Cache and Chat History manager with support for multimodal (Currently tested on image).

## Prerequisites

- **Compiler**: A C++ compiler supporting C++23.
- **CMake**: Version 3.24 or higher.
- **Dependencies**:
  - [spdlog](https://github.com/gabime/spdlog) (Logging)
  - [re2](https://github.com/google/re2) (Regular expression parsing)
  - [llama.cpp](https://github.com/ggerganov/llama.cpp), absolutely

## Building

Currently, this project is designed to be used as a library by adding it to your subdirectory.

```bash
# Clone the repository including submodules
git clone --recursive https://github.com/Bubs2/llama.cpp_wrapper.git

# Build the project
mkdir build
cd build
cmake .. -DGGML_CUDA=ON # Enable CUDA if needed
cmake --build . --config Release
```

## Multimodal Input Format

To include media in your conversation, use the following syntax in the message content:
- **Images/Audio**: `<__path:/absolute/or/relative/path/to/file__>`

Session will automatically parse these tags, load the media via `MTMD`, and manage the associated tokens.

## Quick Start

```cpp
#include "model_server.h"
#include "llama_configs.h"
#include "llama_strategy.h"
#include "llama_inputs.h"
#include "llama_session.h"

#include <string>
#include <iostream>

using namespace llama_server;

int main(int argc, char* argv[]) {

	ModelServer& server = ModelServer::get_server();

    // Load a model
    server.load_model({
        .model_path = "path/to/model.gguf",
        .mtmd_path = "path/to/mmproj.gguf",
        .n_gpu_layers = 99
    }, "my_model");

	// Create a session
	auto session = server.get_session(
		"my_model",
		ContextConfig{
			.n_ctx = 4096,
			.n_batch = 2048,
			.n_ubatch = 2048,
		}
	);

	// Simple estimation is about 3 times faster than Accurate estimation(default)
	session->set_token_estimate_strategy(TokenEstimateStrategy(
		[](const TokenizeCallback& tokenize_cb, std::string_view str) {
			// Simple estimation: 1 token per 2 characters.
			// Not for multimodal
			return (str.size() + 1) / 2;
		},
		64 // Margin
	));

	std::vector<Message> head_msgs;
	std::vector<Message> tail_msgs;
	std::vector<Tool> tools;

    // Chat with multimodal input
    tail_msgs.emplace_back(Message{
        .role = "user",
        .content = "What's in this image? <__path:/path/to/image.jpg__>"
    });

    session->generate(
        head_msgs, tail_msgs, tools,
        GenConfig{
            .max_tokens = 512,
            .stream = true,
            .output_callback = [](std::string&& piece) { std::cout << piece << std::flush; return true; }
    	}
    );
    
    server.shutdown();
}
```

**Note**: For advanced usage like Tool Calling or complex scenarios, please refer to the test/ directory.

## Roadmap
- [x] **Decoupled Architecture**: Separate `HistoryManager` from `LlamaSession` to enable flexible context resizing and independent history management (e.g., switching sessions/models while keeping chat history). Now it has been replaced by `InputEncoder`, which is an internal class.
- [x] ~~**Dynamic History Persistence**: Implement on-disk caching for `HistoryManager` to handle long conversations with minimal RAM usage.~~ (Messages now will be managed by user).
- [x] ~~**Message Editing**: Modify existing messages and tools in the history.~~ (Same as above).
- [ ] **KV Cache Persistence**: Save/load raw KV cache state (Low priority).
