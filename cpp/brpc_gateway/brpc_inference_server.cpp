#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <brpc/controller.h>
#include <brpc/server.h>
#include <butil/time.h>

#include "recommend.pb.h"

namespace {

struct ServerConfig {
  int listen_port = 18100;
  int idle_timeout_sec = -1;
  std::string backend = "semantic_map";
  std::string semantic_map_path = "/app/data/tenrec/processed/semantic_id_map.json";
};

struct Candidate {
  int item_id = 0;
  std::vector<int> semantic_id;
};

bool ConsumeArgValue(const char* arg, const std::string& name, std::string* out) {
  const std::string prefix = "--" + name + "=";
  const std::string value(arg);
  if (value.rfind(prefix, 0) != 0) {
    return false;
  }
  *out = value.substr(prefix.size());
  return true;
}

void PrintUsage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " [options]\n"
      << "  --listen_port=18100\n"
      << "  --backend=semantic_map|trtllm_cpp\n"
      << "  --semantic_map_path=/app/data/tenrec/processed/semantic_id_map.json\n"
      << "  --idle_timeout_sec=-1\n";
}

bool ParseArgs(int argc, char** argv, ServerConfig* config) {
  for (int i = 1; i < argc; ++i) {
    std::string value;
    if (ConsumeArgValue(argv[i], "listen_port", &value)) {
      config->listen_port = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "backend", &value)) {
      config->backend = value;
    } else if (ConsumeArgValue(argv[i], "semantic_map_path", &value)) {
      config->semantic_map_path = value;
    } else if (ConsumeArgValue(argv[i], "idle_timeout_sec", &value)) {
      config->idle_timeout_sec = std::atoi(value.c_str());
    } else if (std::string(argv[i]) == "--help") {
      PrintUsage(argv[0]);
      return false;
    } else {
      std::cerr << "Unknown argument: " << argv[i] << "\n";
      PrintUsage(argv[0]);
      return false;
    }
  }
  return true;
}

std::string SemanticKey(const std::vector<int>& semantic_id) {
  std::ostringstream out;
  for (size_t i = 0; i < semantic_id.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << semantic_id[i];
  }
  return out.str();
}

std::string SemanticKey(const pairec::inference::SemanticId& semantic_id) {
  std::ostringstream out;
  for (int i = 0; i < semantic_id.value_size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << semantic_id.value(i);
  }
  return out.str();
}

uint64_t StableRequestHash(const pairec::inference::RecommendRequest& request) {
  uint64_t h = 1469598103934665603ull;
  auto mix = [&h](uint64_t value) {
    h ^= value;
    h *= 1099511628211ull;
  };
  for (char ch : request.user_id()) {
    mix(static_cast<unsigned char>(ch));
  }
  for (const auto& semantic_id : request.history()) {
    for (int value : semantic_id.value()) {
      mix(static_cast<uint64_t>(value + 1024));
    }
  }
  return h;
}

class SemanticMapParser {
 public:
  explicit SemanticMapParser(std::string text) : text_(std::move(text)) {}

  bool Parse(std::vector<Candidate>* candidates, std::string* error) {
    SkipWs();
    if (!Consume('{')) {
      *error = "semantic map must start with object";
      return false;
    }
    SkipWs();
    if (Consume('}')) {
      return true;
    }
    while (true) {
      std::string item_id_text;
      if (!ParseString(&item_id_text)) {
        *error = "failed to parse item id string";
        return false;
      }
      SkipWs();
      if (!Consume(':')) {
        *error = "expected ':' after item id";
        return false;
      }
      Candidate candidate;
      candidate.item_id = std::atoi(item_id_text.c_str());
      if (!ParseIntArray(&candidate.semantic_id)) {
        *error = "failed to parse semantic id array for item " + item_id_text;
        return false;
      }
      if (!candidate.semantic_id.empty()) {
        candidates->push_back(std::move(candidate));
      }
      SkipWs();
      if (Consume('}')) {
        return true;
      }
      if (!Consume(',')) {
        *error = "expected ',' or '}' after semantic id array";
        return false;
      }
      SkipWs();
    }
  }

 private:
  void SkipWs() {
    while (pos_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
  }

  bool Consume(char expected) {
    SkipWs();
    if (pos_ >= text_.size() || text_[pos_] != expected) {
      return false;
    }
    ++pos_;
    return true;
  }

  bool ParseString(std::string* output) {
    SkipWs();
    if (pos_ >= text_.size() || text_[pos_] != '"') {
      return false;
    }
    ++pos_;
    output->clear();
    while (pos_ < text_.size()) {
      char ch = text_[pos_++];
      if (ch == '"') {
        return true;
      }
      if (ch == '\\') {
        if (pos_ >= text_.size()) {
          return false;
        }
        ch = text_[pos_++];
      }
      output->push_back(ch);
    }
    return false;
  }

  bool ParseInt(int* output) {
    SkipWs();
    bool negative = false;
    if (pos_ < text_.size() && text_[pos_] == '-') {
      negative = true;
      ++pos_;
    }
    if (pos_ >= text_.size() ||
        !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
      return false;
    }
    int value = 0;
    while (pos_ < text_.size() &&
           std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
      value = value * 10 + (text_[pos_] - '0');
      ++pos_;
    }
    *output = negative ? -value : value;
    return true;
  }

  bool ParseIntArray(std::vector<int>* values) {
    values->clear();
    if (!Consume('[')) {
      return false;
    }
    SkipWs();
    if (Consume(']')) {
      return true;
    }
    while (true) {
      int value = 0;
      if (!ParseInt(&value)) {
        return false;
      }
      values->push_back(value);
      SkipWs();
      if (Consume(']')) {
        return true;
      }
      if (!Consume(',')) {
        return false;
      }
    }
  }

  std::string text_;
  size_t pos_ = 0;
};

class InferenceBackend {
 public:
  virtual ~InferenceBackend() = default;
  virtual bool Init(const ServerConfig& config, std::string* error) = 0;
  virtual void Recommend(
      const pairec::inference::RecommendRequest& request,
      pairec::inference::RecommendResponse* response) = 0;
  virtual std::string Name() const = 0;
};

class SemanticMapBackend final : public InferenceBackend {
 public:
  bool Init(const ServerConfig& config, std::string* error) override {
    std::ifstream in(config.semantic_map_path);
    if (!in) {
      *error = "failed to open semantic map: " + config.semantic_map_path;
      return false;
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    SemanticMapParser parser(buffer.str());
    if (!parser.Parse(&candidates_, error)) {
      return false;
    }
    if (candidates_.empty()) {
      *error = "semantic map is empty: " + config.semantic_map_path;
      return false;
    }
    std::sort(candidates_.begin(), candidates_.end(),
              [](const Candidate& lhs, const Candidate& rhs) {
                return lhs.item_id < rhs.item_id;
              });
    std::cout << "[brpc-inference] loaded semantic map entries="
              << candidates_.size() << " path=" << config.semantic_map_path << std::endl;
    return true;
  }

  void Recommend(
      const pairec::inference::RecommendRequest& request,
      pairec::inference::RecommendResponse* response) override {
    std::unordered_set<std::string> history_keys;
    for (const auto& history : request.history()) {
      history_keys.insert(SemanticKey(history));
    }

    const int topk = request.has_topk() && request.topk() > 0 ? request.topk() : 10;
    const size_t offset = StableRequestHash(request) % candidates_.size();
    std::unordered_set<int> emitted_items;

    for (size_t i = 0; i < candidates_.size() && response->recommendations_size() < topk; ++i) {
      const Candidate& candidate = candidates_[(offset + i) % candidates_.size()];
      if (emitted_items.find(candidate.item_id) != emitted_items.end()) {
        continue;
      }
      if (history_keys.find(SemanticKey(candidate.semantic_id)) != history_keys.end()) {
        continue;
      }
      emitted_items.insert(candidate.item_id);
      auto* rec = response->add_recommendations();
      rec->set_item_id(candidate.item_id);
      rec->set_score(1.0);
      for (int value : candidate.semantic_id) {
        rec->add_semantic_id(value);
      }
    }
  }

  std::string Name() const override {
    return "cpp-semantic-map-fallback";
  }

 private:
  std::vector<Candidate> candidates_;
};

class TrtllmCppBackend final : public InferenceBackend {
 public:
  bool Init(const ServerConfig&, std::string* error) override {
    *error = "trtllm_cpp backend is not linked yet; build the TensorRT-LLM C++ "
             "runner adapter inside the ARM TRT-LLM runtime image";
    return false;
  }

  void Recommend(
      const pairec::inference::RecommendRequest&,
      pairec::inference::RecommendResponse*) override {}

  std::string Name() const override {
    return "trtllm_cpp";
  }
};

std::unique_ptr<InferenceBackend> CreateBackend(const std::string& name) {
  if (name == "semantic_map") {
    return std::unique_ptr<InferenceBackend>(new SemanticMapBackend());
  }
  if (name == "trtllm_cpp") {
    return std::unique_ptr<InferenceBackend>(new TrtllmCppBackend());
  }
  return nullptr;
}

class NativeInferenceServiceImpl final : public pairec::inference::RecommendService {
 public:
  explicit NativeInferenceServiceImpl(InferenceBackend* backend) : backend_(backend) {}

  void Recommend(
      google::protobuf::RpcController* controller,
      const pairec::inference::RecommendRequest* request,
      pairec::inference::RecommendResponse* response,
      google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    auto* cntl = static_cast<brpc::Controller*>(controller);
    butil::Timer timer;
    timer.start();

    response->set_code(200);
    response->set_user_id(request->user_id());
    backend_->Recommend(*request, response);

    timer.stop();
    response->set_inference_time_ms(timer.m_elapsed());
    auto* trace = response->mutable_trace();
    trace->set_total_ms(timer.m_elapsed());
    trace->set_infer_ms(timer.m_elapsed());
    trace->set_backend(backend_->Name());

    if (response->recommendations_size() == 0) {
      response->set_code(299);
      response->set_error("items size not enough");
      cntl->SetFailed(response->error());
    }

    std::cout << "[brpc-inference] method=Recommend user=" << request->user_id()
              << " code=" << response->code()
              << " items=" << response->recommendations_size()
              << " latency_ms=" << timer.m_elapsed()
              << " backend=" << backend_->Name() << std::endl;
  }

  void Health(
      google::protobuf::RpcController*,
      const pairec::inference::HealthRequest*,
      pairec::inference::HealthResponse* response,
      google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    response->set_code(200);
    response->set_status("healthy");
    response->set_backend(backend_->Name());
  }

 private:
  InferenceBackend* backend_;
};

}  // namespace

int main(int argc, char** argv) {
  ServerConfig config;
  if (!ParseArgs(argc, argv, &config)) {
    return 2;
  }

  std::unique_ptr<InferenceBackend> backend = CreateBackend(config.backend);
  if (!backend) {
    std::cerr << "Unsupported backend: " << config.backend << std::endl;
    return 2;
  }

  std::string error;
  if (!backend->Init(config, &error)) {
    std::cerr << "Failed to initialize backend " << config.backend
              << ": " << error << std::endl;
    return 1;
  }

  NativeInferenceServiceImpl service(backend.get());
  brpc::Server server;
  if (server.AddService(&service, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
    std::cerr << "Failed to add RecommendService" << std::endl;
    return 1;
  }

  brpc::ServerOptions options;
  options.idle_timeout_sec = config.idle_timeout_sec;
  if (server.Start(config.listen_port, &options) != 0) {
    std::cerr << "Failed to start brpc inference server on port "
              << config.listen_port << std::endl;
    return 1;
  }

  std::cout << "brpc inference server listening on 0.0.0.0:" << config.listen_port
            << ", backend=" << backend->Name() << std::endl;
  server.RunUntilAskedToQuit();
  return 0;
}
