# llama_server_cpp

A lightweight C++23 inference engine built on top of [llama.cpp](https://github.com/ggerganov/llama.cpp). This library works as a wrapper and a basic KV Cache and Chat History manager with support for multi-model (Currently tested on image).

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
git clone --recursive https://github.com/Bubs2/llama_server_cpp.git

# Build the project
mkdir build
cd build
cmake .. -DGGML_CUDA=ON # Enable CUDA if needed
cmake --build . --config Release
```

## Multimodal Input Format

To include media in your conversation, use the following syntax in the message content:
- **Images/Audio**: `<__path:/absolute/or/relative/path/to/file__>`

The `HistoryManager` will automatically parse these tags, load the media via `MTMD`, and manage the associated tokens.

## Quick Start

```cpp
#include "model_server.h"
#include "llama_configs.h"
#include "llama_session.h"
#include "history_manager.h"

using namespace llama_server;

int main() {
    auto& server = ModelServer::get_server();
    
    // Load a model
    server.load_model({
        .model_path = "path/to/model.gguf",
        .mtmd_path = "path/to/mmproj.gguf",
        .n_gpu_layers = 99
    }, "my_model");

    // Create a session
    auto session = server.get_session("my_model", {.n_ctx = 4096});

    // Chat with multimodal input
    session->access_history_manager().add_message({
        .role = "user",
        .content = "What's in this image? <__path:/path/to/image.jpg__>"
    });

    session->generate({
        .max_tokens = 512,
        .stream = true,
        .output_callback = [](std::string piece) { std::cout << piece << std::flush; }
    });
    
    server.shutdown();
}
```

**Note**: For advanced usage like Tool Calling or complex scenarios, please refer to the test/ directory.
