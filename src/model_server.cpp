#include "model_server.h"
#include "llama_model.h"
#include "llama_log.h"
#include "llama.h"
#include "mtmd.h"

#include <format>

namespace llama_server {

    using namespace internal;
    using namespace model_server_details;

    namespace model_server_details {
        void check_model_available(const LlamaModel* model) {
            const auto* vocab = model->get_vocab();
            
            if (
                (llama_vocab_bos(vocab) == LLAMA_TOKEN_NULL || !llama_vocab_is_control(vocab, llama_vocab_bos(vocab))) &&
                (llama_vocab_eos(vocab) == LLAMA_TOKEN_NULL || !llama_vocab_is_control(vocab, llama_vocab_eos(vocab)))
            ) throw LlamaException("Model has neither a BOS nor an EOS token.");
        }
    }

    // ===================================================================
    // ModelServer
    // ===================================================================

    ModelServer::ModelServer() {}
    ModelServer::~ModelServer() {}

    ModelServer& ModelServer::get_server() {
        static ModelServer server = ModelServer();
        return server;
    }

    void ModelServer::shutdown() {
        if (shutdown_flag_) return;

        shutdown_flag_ = true;

        std::unique_lock lock(mutex_);

        auto delete_queue = std::move(model_map_);
        model_map_.clear();
        loading_model_set_.clear();

        lock.unlock();
        loading_model_cv_.notify_all();

        delete_queue.clear();
    }

    void ModelServer::load_model(
        const ModelConfig& config,
        std::string name
    ) {
        if (shutdown_flag_) throw ServerShutdownException("ModelServer is shutdown. Cannot load model: " + name);

        std::unique_lock lock(mutex_);

        if (shutdown_flag_) throw ServerShutdownException("ModelServer is shutdown. Cannot load model: " + name);

        if (model_map_.find(name) != model_map_.end()) return;

        if (loading_model_set_.find(name) != loading_model_set_.end()) while (true) {
            log_info(std::format("ModelServer: Waiting for model loading: {}", name));
            loading_model_cv_.wait(lock);
            if (loading_model_set_.find(name) == loading_model_set_.end()) {
                if (model_map_.find(name) == model_map_.end()) throw LlamaException(std::format("ModelServer: Model loading failed for some reason: {}", name));
                return;
            }
        }

        loading_model_set_.emplace(name);

        lock.unlock();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = config.n_gpu_layers;
        model_params.use_mmap = config.use_mmap;
        model_params.use_mlock = config.use_mlock;

        mtmd_context_params mtmd_params = mtmd_context_params_default();
        mtmd_params.image_min_tokens = config.image_min_tokens;
        mtmd_params.image_max_tokens = config.image_max_tokens;

        std::shared_ptr<LlamaModel> model;

        try { model = std::make_shared<LlamaModel>(config.model_path, model_params, config.mtmd_path, mtmd_params); }
        catch (const std::exception& e) { // Catch all exceptions
            log_error(e.what());
            lock.lock();
            loading_model_set_.erase(name);
            lock.unlock();
            loading_model_cv_.notify_all();
            throw LlamaException("ModelServer: Failed to load model: " + name);
        }

        try { check_model_available(model.get()); }
        catch (const LlamaException& e) {
            lock.lock();
            loading_model_set_.erase(name);
            lock.unlock();
            loading_model_cv_.notify_all();
            throw LlamaException("ModelServer: Model not supported: " + name + ". " + e.what());
        }

        lock.lock();

        if (shutdown_flag_) {
            loading_model_set_.erase(name);
            lock.unlock();
            loading_model_cv_.notify_all();
            throw ServerShutdownException("ModelServer is shutdown while loading model: " + name);
        }

        loading_model_set_.erase(name);
        model_map_.emplace(std::move(name), std::move(model));
        lock.unlock();
        loading_model_cv_.notify_all();
    }

    void ModelServer::unload_model(
        std::string name
    ) {
        if (shutdown_flag_) throw ServerShutdownException("ModelServer is shutdown. Cannot load model: " + name);

        std::unique_lock lock(mutex_);

        if (shutdown_flag_) throw ServerShutdownException("ModelServer is shutdown. Cannot get model: " + name);

        while (loading_model_set_.find(name) != loading_model_set_.end()) loading_model_cv_.wait(lock);

        model_map_.erase(name);
    }

    std::unique_ptr<LlamaSession> ModelServer::get_session(
        std::string model_name,
        ContextConfig context_config
    ) const {
        std::shared_lock lock(mutex_);

        if (shutdown_flag_) throw ServerShutdownException("ModelServer is shutdown. Cannot get session.");

        while (loading_model_set_.find(model_name) != loading_model_set_.end()) {
            log_info(std::format("ModelServer: Waiting for model loading: {}", model_name));
            loading_model_cv_.wait(lock);

            if (shutdown_flag_) throw ServerShutdownException("ModelServer shutdown while loading model: " + model_name);
        }

        auto model = model_map_.find(model_name);
        if (model == model_map_.end()) {
            throw LlamaException("Model not found: " + model_name);
        }

        return std::make_unique<LlamaSession>(context_config, model->second);
    }

}