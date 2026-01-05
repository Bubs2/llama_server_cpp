#pragma once

#include <string>

namespace llama_server {
    struct Message {
        std::string role;
        std::string content;
    };

    struct Tool {
        std::string name;
        std::string description;
        std::string parameters;
    };
}