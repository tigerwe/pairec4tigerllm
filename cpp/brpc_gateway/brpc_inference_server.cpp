#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <brpc/controller.h>
#include <brpc/server.h>
#include <butil/time.h>

#include "recommend.pb.h"

#if defined(PAIREC_ENABLE_DATASYSTEM_KV_PROBE) && PAIREC_ENABLE_DATASYSTEM_KV_PROBE
#include <datasystem/kv_client.h>
#include <datasystem/utils/connection.h>
#endif

#if defined(PAIREC_ENABLE_TRTLLM_CPP) && PAIREC_ENABLE_TRTLLM_CPP
#include <NvInferRuntime.h>
#include <tensorrt_llm/executor/executor.h>
#include <tensorrt_llm/executor/types.h>
#include <tensorrt_llm/plugins/api/tllmPlugin.h>
#endif

namespace {

struct ServerConfig {
  int listen_port = 18100;
  int idle_timeout_sec = -1;
  std::string backend = "semantic_map";
  std::string semantic_map_path = "/app/data/tenrec/processed/semantic_id_map.json";
  std::string trt_engine_dir = "/app/trt_engines/qwen3_rec_v4";
  std::string trt_tokenizer_config_path =
      "/app/exported/qwen3_rec/pairec_cpp_tokenizer.txt";
  int trt_max_input_len = 64;
  int trt_max_new_tokens = 32;
  int trt_num_samples = 1;
  int trt_top_k = 50;
  double trt_temperature = 0.7;
  int trt_max_batch_size = 1;
  int trt_max_num_tokens = 96;
  int trt_max_kv_tokens = 256;
  size_t trt_kv_cache_host_cache_size = 0;
  std::string trt_scheduler_policy = "guaranteed_no_evict";
  int trt_request_timeout_ms = 30000;
  bool trt_plugin_preflight_only = false;
  bool trt_datasystem_mget_probe = false;
  std::string datasystem_host;
  int datasystem_port = 0;
  int kvc_probe_object_count = 4;
  uint64_t kvc_probe_object_bytes = 3670016;
  int kvc_probe_timeout_ms = 5000;
  int kvc_probe_ttl_sec = 120;
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
      << "  --backend=semantic_map|trtllm_cpp|datasystem_kv_probe\n"
      << "  --semantic_map_path=/app/data/tenrec/processed/semantic_id_map.json\n"
      << "  --trt_engine_dir=/app/trt_engines/qwen3_rec_v4\n"
      << "  --trt_tokenizer_config_path=/app/exported/qwen3_rec/pairec_cpp_tokenizer.txt\n"
      << "  --trt_max_input_len=64\n"
      << "  --trt_max_new_tokens=32\n"
      << "  --trt_num_samples=1\n"
      << "  --trt_top_k=50\n"
      << "  --trt_temperature=0.7\n"
      << "  --trt_max_batch_size=1\n"
      << "  --trt_max_num_tokens=96\n"
      << "  --trt_max_kv_tokens=256\n"
      << "  --trt_kv_cache_host_cache_size=0\n"
      << "  --trt_scheduler_policy=guaranteed_no_evict|max_utilization\n"
      << "  --trt_request_timeout_ms=30000\n"
      << "  --trt_plugin_preflight_only=0\n"
      << "  --trt_datasystem_mget_probe=0\n"
      << "  --datasystem_host=141.61.91.189\n"
      << "  --datasystem_port=18481\n"
      << "  --kvc_probe_object_count=4\n"
      << "  --kvc_probe_object_bytes=3670016\n"
      << "  --kvc_probe_timeout_ms=5000\n"
      << "  --kvc_probe_ttl_sec=120\n"
      << "  --idle_timeout_sec=-1\n";
}

bool ParseBoolFlag(const std::string& value) {
  return value == "1" || value == "true" || value == "TRUE" ||
         value == "on" || value == "ON" || value == "yes" ||
         value == "YES";
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
    } else if (ConsumeArgValue(argv[i], "trt_engine_dir", &value)) {
      config->trt_engine_dir = value;
    } else if (ConsumeArgValue(argv[i], "trt_tokenizer_config_path", &value)) {
      config->trt_tokenizer_config_path = value;
    } else if (ConsumeArgValue(argv[i], "trt_max_input_len", &value)) {
      config->trt_max_input_len = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "trt_max_new_tokens", &value)) {
      config->trt_max_new_tokens = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "trt_num_samples", &value)) {
      config->trt_num_samples = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "trt_top_k", &value)) {
      config->trt_top_k = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "trt_temperature", &value)) {
      config->trt_temperature = std::atof(value.c_str());
    } else if (ConsumeArgValue(argv[i], "trt_max_batch_size", &value)) {
      config->trt_max_batch_size = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "trt_max_num_tokens", &value)) {
      config->trt_max_num_tokens = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "trt_max_kv_tokens", &value)) {
      config->trt_max_kv_tokens = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "trt_kv_cache_host_cache_size", &value)) {
      config->trt_kv_cache_host_cache_size =
          static_cast<size_t>(std::strtoull(value.c_str(), nullptr, 10));
    } else if (ConsumeArgValue(argv[i], "trt_scheduler_policy", &value)) {
      config->trt_scheduler_policy = value;
    } else if (ConsumeArgValue(argv[i], "trt_request_timeout_ms", &value)) {
      config->trt_request_timeout_ms = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "trt_plugin_preflight_only", &value)) {
      config->trt_plugin_preflight_only = ParseBoolFlag(value);
    } else if (std::string(argv[i]) == "--trt_plugin_preflight_only") {
      config->trt_plugin_preflight_only = true;
    } else if (ConsumeArgValue(argv[i], "trt_datasystem_mget_probe", &value)) {
      config->trt_datasystem_mget_probe = ParseBoolFlag(value);
    } else if (std::string(argv[i]) == "--trt_datasystem_mget_probe") {
      config->trt_datasystem_mget_probe = true;
    } else if (ConsumeArgValue(argv[i], "datasystem_host", &value)) {
      config->datasystem_host = value;
    } else if (ConsumeArgValue(argv[i], "datasystem_port", &value)) {
      config->datasystem_port = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "kvc_probe_object_count", &value)) {
      config->kvc_probe_object_count = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "kvc_probe_object_bytes", &value)) {
      config->kvc_probe_object_bytes =
          static_cast<uint64_t>(std::strtoull(value.c_str(), nullptr, 10));
    } else if (ConsumeArgValue(argv[i], "kvc_probe_timeout_ms", &value)) {
      config->kvc_probe_timeout_ms = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "kvc_probe_ttl_sec", &value)) {
      config->kvc_probe_ttl_sec = std::atoi(value.c_str());
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

std::string Trim(const std::string& input) {
  size_t begin = 0;
  while (begin < input.size() &&
         std::isspace(static_cast<unsigned char>(input[begin]))) {
    ++begin;
  }
  size_t end = input.size();
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(input[end - 1]))) {
    --end;
  }
  return input.substr(begin, end - begin);
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

#if defined(PAIREC_ENABLE_TRTLLM_CPP) && PAIREC_ENABLE_TRTLLM_CPP
bool RunTrtLlmPluginPreflight(std::string* error) {
  if (!initTrtLlmPlugins()) {
    *error = "failed to initialize TensorRT-LLM plugins";
    return false;
  }

  std::int32_t legacy_creator_count = 0;
  auto legacy_creators = getPluginCreators(legacy_creator_count);
  std::int32_t v3_creator_count = 0;
  getCreators(v3_creator_count);

  std::ostringstream names;
  std::unordered_set<std::string> creator_names;
  for (std::int32_t i = 0; i < legacy_creator_count; ++i) {
    const auto* creator = legacy_creators[i];
    if (creator == nullptr) {
      continue;
    }
    const std::string name = creator->getPluginName();
    creator_names.insert(name);
    if (i != 0) {
      names << ",";
    }
    names << name << ":v" << creator->getPluginVersion();
  }

  const int total_creator_count = legacy_creator_count + v3_creator_count;
  std::cout << "[brpc-inference] trtllm plugin preflight ok"
            << " namespace=tensorrt_llm"
            << " legacy_creators=" << legacy_creator_count
            << " v3_creators=" << v3_creator_count
            << " creators=" << names.str()
            << std::endl;

  if (total_creator_count <= 0) {
    *error = "TensorRT-LLM plugin library registered zero plugin creators";
    return false;
  }

  std::vector<std::string> missing_required;
  const char* required_creators[] = {"Gemm", "GPTAttention", "GemmSwiglu"};
  for (const char* required : required_creators) {
    if (creator_names.find(required) == creator_names.end()) {
      missing_required.emplace_back(required);
    }
  }
  if (!missing_required.empty()) {
    std::ostringstream missing;
    for (size_t i = 0; i < missing_required.size(); ++i) {
      if (i != 0) {
        missing << ",";
      }
      missing << missing_required[i];
    }
    *error = "TensorRT-LLM plugin registry is missing required creators: " +
             missing.str();
    return false;
  }
  return true;
}
#else
bool RunTrtLlmPluginPreflight(std::string* error) {
  *error = "trtllm_cpp backend is not linked; rebuild with "
           "-DPAIREC_ENABLE_TRTLLM_CPP=ON";
  return false;
}
#endif

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

bool LoadSemanticCandidates(
    const std::string& path,
    std::vector<Candidate>* candidates,
    std::string* error) {
  std::ifstream in(path);
  if (!in) {
    *error = "failed to open semantic map: " + path;
    return false;
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  SemanticMapParser parser(buffer.str());
  if (!parser.Parse(candidates, error)) {
    return false;
  }
  if (candidates->empty()) {
    *error = "semantic map is empty: " + path;
    return false;
  }
  std::sort(candidates->begin(), candidates->end(),
            [](const Candidate& lhs, const Candidate& rhs) {
              return lhs.item_id < rhs.item_id;
            });
  return true;
}

struct CppTokenizerConfig {
  int eos_id = -1;
  int pad_id = -1;
  std::vector<int> prefix_tokens;
  std::vector<int> separator_tokens;
  std::vector<int> suffix_tokens;
  std::vector<std::vector<int>> semantic_token_ids;
  std::unordered_map<int, std::pair<int, int>> id_to_semantic;

  bool Load(const std::string& path, std::string* error) {
    std::ifstream in(path);
    if (!in) {
      *error = "failed to open C++ tokenizer config: " + path;
      return false;
    }

    semantic_token_ids.assign(4, std::vector<int>(256, -1));
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
      ++line_no;
      line = Trim(line);
      if (line.empty() || line[0] == '#') {
        continue;
      }
      std::istringstream iss(line);
      std::string key;
      iss >> key;
      if (key == "eos_id") {
        iss >> eos_id;
      } else if (key == "pad_id") {
        iss >> pad_id;
      } else if (key == "prefix") {
        if (!ParseTokenList(iss, &prefix_tokens)) {
          *error = "failed to parse prefix at line " + std::to_string(line_no);
          return false;
        }
      } else if (key == "separator") {
        if (!ParseTokenList(iss, &separator_tokens)) {
          *error = "failed to parse separator at line " + std::to_string(line_no);
          return false;
        }
      } else if (key == "suffix") {
        if (!ParseTokenList(iss, &suffix_tokens)) {
          *error = "failed to parse suffix at line " + std::to_string(line_no);
          return false;
        }
      } else if (key == "semantic") {
        int layer = -1;
        int value = -1;
        int token_id = -1;
        iss >> layer >> value >> token_id;
        if (layer < 0 || value < 0 || token_id < 0) {
          *error = "invalid semantic token entry at line " + std::to_string(line_no);
          return false;
        }
        if (static_cast<size_t>(layer) >= semantic_token_ids.size()) {
          semantic_token_ids.resize(layer + 1);
        }
        if (static_cast<size_t>(value) >= semantic_token_ids[layer].size()) {
          semantic_token_ids[layer].resize(value + 1, -1);
        }
        semantic_token_ids[layer][value] = token_id;
        id_to_semantic[token_id] = std::make_pair(layer, value);
      } else {
        *error = "unknown tokenizer config key at line " +
                 std::to_string(line_no) + ": " + key;
        return false;
      }
    }

    if (eos_id < 0 || pad_id < 0) {
      *error = "tokenizer config must define eos_id and pad_id";
      return false;
    }
    if (prefix_tokens.empty() || suffix_tokens.empty()) {
      *error = "tokenizer config must define non-empty prefix and suffix";
      return false;
    }
    if (semantic_token_ids.size() < 4 || id_to_semantic.empty()) {
      *error = "tokenizer config has no semantic token mappings";
      return false;
    }
    return true;
  }

 private:
  static bool ParseTokenList(std::istringstream& iss, std::vector<int>* tokens) {
    tokens->clear();
    int token = 0;
    while (iss >> token) {
      tokens->push_back(token);
    }
    return !tokens->empty();
  }
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
    if (!LoadSemanticCandidates(config.semantic_map_path, &candidates_, error)) {
      return false;
    }
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

class DataSystemKvProbeBackend final : public InferenceBackend {
 public:
#if defined(PAIREC_ENABLE_DATASYSTEM_KV_PROBE) && PAIREC_ENABLE_DATASYSTEM_KV_PROBE
  bool Init(const ServerConfig& config, std::string* error) override {
    config_ = config;
    if (config_.kvc_probe_object_count < 1) {
      *error = "kvc_probe_object_count must be >= 1";
      return false;
    }
    if (config_.kvc_probe_object_bytes == 0) {
      *error = "kvc_probe_object_bytes must be > 0";
      return false;
    }

    const char* env_host = std::getenv("DATASYSTEM_HOST");
    const char* env_port = std::getenv("DATASYSTEM_PORT");
    if (config_.datasystem_host.empty() && env_host != nullptr) {
      config_.datasystem_host = env_host;
    }
    if (config_.datasystem_port <= 0 && env_port != nullptr) {
      config_.datasystem_port = std::atoi(env_port);
    }
    if (config_.datasystem_host.empty()) {
      config_.datasystem_host = "127.0.0.1";
    }
    if (config_.datasystem_port <= 0) {
      config_.datasystem_port = 31501;
    }

    datasystem::ConnectOptions options;
    options.host = config_.datasystem_host;
    options.port = config_.datasystem_port;
    options.enableCrossNodeConnection = true;
    kv_client_.reset(new datasystem::KVClient(options));
    datasystem::Status init_status = kv_client_->Init();
    if (init_status.IsError()) {
      *error = "failed to initialize DataSystem KVClient: " + init_status.ToString();
      return false;
    }

    std::cout << "[brpc-inference] datasystem_kv_probe initialized"
              << " host=" << config_.datasystem_host
              << " port=" << config_.datasystem_port
              << " object_count=" << config_.kvc_probe_object_count
              << " object_bytes=" << config_.kvc_probe_object_bytes
              << " ttl_sec=" << config_.kvc_probe_ttl_sec
              << " timeout_ms=" << config_.kvc_probe_timeout_ms
              << std::endl;
    return true;
  }

  void Recommend(
      const pairec::inference::RecommendRequest& request,
      pairec::inference::RecommendResponse* response) override {
    std::vector<std::string> keys = BuildKeys(request);
    std::vector<uint64_t> sizes(
        keys.size(), static_cast<uint64_t>(config_.kvc_probe_object_bytes));
    std::vector<std::shared_ptr<datasystem::Buffer>> buffers;

    datasystem::SetParam set_param;
    set_param.writeMode = datasystem::WriteMode::NONE_L2_CACHE_EVICT;
    set_param.ttlSecond = static_cast<uint32_t>(std::max(0, config_.kvc_probe_ttl_sec));
    set_param.existence = datasystem::ExistenceOpt::NONE;
    set_param.cacheType = datasystem::CacheType::MEMORY;

    butil::Timer mcreate_timer;
    mcreate_timer.start();
    datasystem::Status create_status =
        kv_client_->MCreate(keys, sizes, set_param, buffers);
    mcreate_timer.stop();
    if (create_status.IsError()) {
      response->set_error("DataSystem MCreate failed: " + create_status.ToString());
      return;
    }
    if (buffers.size() != keys.size()) {
      response->set_error("DataSystem MCreate returned unexpected buffer count");
      return;
    }

    butil::Timer fill_timer;
    fill_timer.start();
    std::vector<char> payload(static_cast<size_t>(config_.kvc_probe_object_bytes));
    for (size_t i = 0; i < payload.size(); ++i) {
      payload[i] = static_cast<char>((i + request_sequence_.load()) & 0xff);
    }
    for (auto& buffer : buffers) {
      datasystem::Status copy_status =
          buffer->MemoryCopy(payload.data(), static_cast<uint64_t>(payload.size()));
      if (copy_status.IsError()) {
        response->set_error("DataSystem probe buffer copy failed: " +
                            copy_status.ToString());
        return;
      }
    }
    fill_timer.stop();

    butil::Timer mset_timer;
    mset_timer.start();
    datasystem::Status set_status = kv_client_->MSet(buffers);
    mset_timer.stop();
    if (set_status.IsError()) {
      response->set_error("DataSystem MSet failed: " + set_status.ToString());
      return;
    }

    butil::Timer mget_timer;
    mget_timer.start();
    std::vector<datasystem::Optional<datasystem::Buffer>> out_buffers;
    datasystem::Status get_status =
        kv_client_->Get(keys, out_buffers, config_.kvc_probe_timeout_ms);
    mget_timer.stop();
    if (get_status.IsError()) {
      response->set_error("DataSystem MGet failed: " + get_status.ToString());
      return;
    }

    size_t found_count = 0;
    uint64_t found_bytes = 0;
    for (const auto& buffer : out_buffers) {
      if (!buffer) {
        continue;
      }
      ++found_count;
      found_bytes += static_cast<uint64_t>(buffer->GetSize());
    }
    if (found_count != keys.size()) {
      std::ostringstream out;
      out << "DataSystem MGet returned " << found_count << "/" << keys.size()
          << " buffers";
      response->set_error(out.str());
      return;
    }

    auto* trace = response->mutable_trace();
    trace->set_kv_write_ms(
        mcreate_timer.m_elapsed() + fill_timer.m_elapsed() + mset_timer.m_elapsed());
    trace->set_kv_lookup_ms(mget_timer.m_elapsed());
    trace->set_kv_source("datasystem_mset_mget_probe");
    trace->set_backend_total_ms(trace->kv_write_ms() + trace->kv_lookup_ms());
    trace->set_generate_ms(trace->backend_total_ms());

    auto* rec = response->add_recommendations();
    rec->set_item_id(1);
    rec->set_score(1.0);
    rec->add_semantic_id(0);
    rec->add_semantic_id(0);
    rec->add_semantic_id(0);
    rec->add_semantic_id(0);

    std::cout << "[brpc-inference] method=KvcMSetMGetProbe"
              << " request_id=" << request.request_id()
              << " user=" << request.user_id()
              << " object_count=" << keys.size()
              << " set_buffer_count=" << buffers.size()
              << " get_key_count=" << keys.size()
              << " object_bytes=" << config_.kvc_probe_object_bytes
              << " total_bytes=" << found_bytes
              << " mcreate_ms=" << mcreate_timer.m_elapsed()
              << " fill_ms=" << fill_timer.m_elapsed()
              << " mset_ms=" << mset_timer.m_elapsed()
              << " mget_ms=" << mget_timer.m_elapsed()
              << " found=" << found_count
              << std::endl;
  }

  std::string Name() const override {
    return "datasystem_kv_probe";
  }

 private:
  static std::string SanitizeKeyComponent(const std::string& input) {
    std::string out;
    out.reserve(std::min<size_t>(input.size(), 96));
    for (char ch : input) {
      const unsigned char c = static_cast<unsigned char>(ch);
      if (std::isalnum(c) || ch == '~' || ch == '!' || ch == '@' ||
          ch == '#' || ch == '$' || ch == '%' || ch == '^' || ch == '&' ||
          ch == '*' || ch == '.' || ch == '-' || ch == '_') {
        out.push_back(ch);
      } else {
        out.push_back('_');
      }
      if (out.size() >= 96) {
        break;
      }
    }
    if (out.empty()) {
      out = "empty";
    }
    return out;
  }

  std::vector<std::string> BuildKeys(
      const pairec::inference::RecommendRequest& request) {
    const uint64_t seq = request_sequence_.fetch_add(1);
    std::string component = request.request_id();
    if (component.empty()) {
      component = request.user_id();
    }
    component = SanitizeKeyComponent(component);

    std::vector<std::string> keys;
    keys.reserve(static_cast<size_t>(config_.kvc_probe_object_count));
    for (int i = 0; i < config_.kvc_probe_object_count; ++i) {
      std::ostringstream out;
      out << "pairec_mget_probe_" << component << "_" << seq << "_" << i;
      keys.push_back(out.str());
    }
    return keys;
  }

  ServerConfig config_;
  std::unique_ptr<datasystem::KVClient> kv_client_;
  std::atomic<uint64_t> request_sequence_{1};
#else
  bool Init(const ServerConfig&, std::string* error) override {
    *error = "datasystem_kv_probe backend is not linked; rebuild with "
             "-DPAIREC_ENABLE_DATASYSTEM_KV_PROBE=ON";
    return false;
  }

  void Recommend(
      const pairec::inference::RecommendRequest&,
      pairec::inference::RecommendResponse*) override {}

  std::string Name() const override {
    return "datasystem_kv_probe";
  }
#endif
};

class TrtllmCppBackend final : public InferenceBackend {
 public:
#if defined(PAIREC_ENABLE_TRTLLM_CPP) && PAIREC_ENABLE_TRTLLM_CPP
  bool Init(const ServerConfig& config, std::string* error) override {
    config_ = config;
    if (config_.trt_num_samples < 1) {
      config_.trt_num_samples = 1;
    }
    if (config_.trt_max_new_tokens < 1) {
      config_.trt_max_new_tokens = 32;
    }
    if (config_.trt_max_input_len < 1) {
      config_.trt_max_input_len = 64;
    }

    if (!tokenizer_.Load(config_.trt_tokenizer_config_path, error)) {
      return false;
    }
    if (!LoadSemanticCandidates(config_.semantic_map_path, &candidates_, error)) {
      return false;
    }
    for (const auto& candidate : candidates_) {
      semantic_to_candidate_[SemanticKey(candidate.semantic_id)] = &candidate;
    }

    try {
      if (!RunTrtLlmPluginPreflight(error)) {
        return false;
      }

      namespace texec = tensorrt_llm::executor;
      texec::SchedulerConfig scheduler_config(ParseSchedulerPolicy(config_.trt_scheduler_policy));
      std::optional<texec::SizeType32> max_kv_tokens = std::nullopt;
      if (config_.trt_max_kv_tokens > 0) {
        max_kv_tokens = static_cast<texec::SizeType32>(config_.trt_max_kv_tokens);
      }
      std::optional<size_t> host_cache_size = std::nullopt;
      if (config_.trt_kv_cache_host_cache_size > 0) {
        host_cache_size = config_.trt_kv_cache_host_cache_size;
      }
      texec::KvCacheConfig kv_cache_config(
          true,
          max_kv_tokens,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          host_cache_size,
          true);
      texec::ExecutorConfig executor_config(1, scheduler_config, kv_cache_config);
      if (config_.trt_max_batch_size > 0) {
        executor_config.setMaxBatchSize(
            static_cast<texec::SizeType32>(config_.trt_max_batch_size));
      }
      if (config_.trt_max_num_tokens > 0) {
        executor_config.setMaxNumTokens(
            static_cast<texec::SizeType32>(config_.trt_max_num_tokens));
      }
      executor_.reset(new texec::Executor(
          std::filesystem::path(config_.trt_engine_dir),
          texec::ModelType::kDECODER_ONLY,
          executor_config));
    } catch (const std::exception& e) {
      *error = std::string("failed to initialize TensorRT-LLM Executor: ") + e.what();
      return false;
    }

#if defined(PAIREC_ENABLE_DATASYSTEM_KV_PROBE) && PAIREC_ENABLE_DATASYSTEM_KV_PROBE
    if (config_.trt_datasystem_mget_probe &&
        !InitDatasystemSetGetProbe(error)) {
      return false;
    }
#else
    if (config_.trt_datasystem_mget_probe) {
      *error = "trt_datasystem_mget_probe requires DataSystem support; rebuild with "
               "-DPAIREC_ENABLE_DATASYSTEM_KV_PROBE=ON";
      return false;
    }
#endif

    std::cout << "[brpc-inference] trtllm_cpp initialized"
              << " engine_dir=" << config_.trt_engine_dir
              << " tokenizer_config=" << config_.trt_tokenizer_config_path
              << " semantic_map_entries=" << candidates_.size()
              << " semantic_token_ids=" << tokenizer_.id_to_semantic.size()
              << " max_input_len=" << config_.trt_max_input_len
              << " max_new_tokens=" << config_.trt_max_new_tokens
              << " max_batch_size=" << config_.trt_max_batch_size
              << " max_num_tokens=" << config_.trt_max_num_tokens
              << " num_samples=" << config_.trt_num_samples
              << " datasystem_mget_probe=" << (config_.trt_datasystem_mget_probe ? 1 : 0)
              << std::endl;
    return true;
  }

  void Recommend(
      const pairec::inference::RecommendRequest& request,
      pairec::inference::RecommendResponse* response) override {
    auto* trace = response->mutable_trace();
    butil::Timer prompt_timer;
    prompt_timer.start();
    std::vector<int> prompt_tokens;
    std::string prompt_error;
    if (!BuildPromptTokens(request, &prompt_tokens, &prompt_error)) {
      response->set_error(prompt_error);
      return;
    }
    prompt_timer.stop();
    trace->set_prompt_ms(prompt_timer.m_elapsed());

    std::vector<int> sampled_tokens;
    double runner_total_ms = 0.0;
    double runner_max_ms = 0.0;
    int runner_calls = 0;
    const uint64_t base_seed =
        seed_.fetch_add(static_cast<uint64_t>(config_.trt_num_samples));
    for (int sample = 0; sample < config_.trt_num_samples; ++sample) {
      std::vector<int> output_tokens;
      double runner_ms = 0.0;
      std::string error;
      if (!RunExecutor(
              prompt_tokens,
              base_seed + static_cast<uint64_t>(sample),
              &output_tokens,
              &runner_ms,
              &error)) {
        response->set_error(error);
        return;
      }
      runner_total_ms += runner_ms;
      runner_max_ms = std::max(runner_max_ms, runner_ms);
      ++runner_calls;
      sampled_tokens.insert(sampled_tokens.end(), output_tokens.begin(), output_tokens.end());
    }
    trace->set_runner_generate_ms(runner_total_ms);
    trace->set_runner_calls(runner_calls);
    if (runner_calls > 0) {
      trace->set_runner_avg_ms(runner_total_ms / runner_calls);
    }
    trace->set_runner_max_ms(runner_max_ms);

    butil::Timer parse_timer;
    parse_timer.start();
    std::vector<std::vector<int>> semantic_candidates = ParseOutputTokens(sampled_tokens);
    parse_timer.stop();
    trace->set_parse_combo_ms(parse_timer.m_elapsed());

    butil::Timer map_timer;
    map_timer.start();
    FillRecommendations(request, semantic_candidates, response);
    map_timer.stop();
    trace->set_map_item_ms(map_timer.m_elapsed());

    if (config_.trt_datasystem_mget_probe) {
#if defined(PAIREC_ENABLE_DATASYSTEM_KV_PROBE) && PAIREC_ENABLE_DATASYSTEM_KV_PROBE
      std::string kv_error;
      if (!RunDatasystemSetGetProbe(request, trace, &kv_error)) {
        response->clear_recommendations();
        response->set_error(kv_error);
        return;
      }
#else
      response->clear_recommendations();
      response->set_error(
          "trt_datasystem_mget_probe requires PAIREC_ENABLE_DATASYSTEM_KV_PROBE");
      return;
#endif
    }

    trace->set_backend_total_ms(
        trace->prompt_ms() + trace->runner_generate_ms() +
        trace->parse_combo_ms() + trace->map_item_ms() +
        trace->kv_write_ms() + trace->kv_lookup_ms());
    trace->set_generate_ms(trace->backend_total_ms());
  }

  std::string Name() const override {
    return "trtllm_cpp";
  }

 private:
  static tensorrt_llm::executor::CapacitySchedulerPolicy ParseSchedulerPolicy(
      std::string policy) {
    std::transform(policy.begin(), policy.end(), policy.begin(),
                   [](unsigned char ch) { return std::tolower(ch); });
    if (policy == "max_utilization" || policy == "max-utilization" ||
        policy == "max_util" || policy == "max") {
      return tensorrt_llm::executor::CapacitySchedulerPolicy::kMAX_UTILIZATION;
    }
    return tensorrt_llm::executor::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT;
  }

  bool BuildPromptTokens(
      const pairec::inference::RecommendRequest& request,
      std::vector<int>* prompt_tokens,
      std::string* error) const {
    std::vector<std::vector<int>> history;
    for (const auto& semantic_id : request.history()) {
      std::vector<int> item;
      bool all_zero = true;
      for (int value : semantic_id.value()) {
        item.push_back(value);
        if (value != 0) {
          all_zero = false;
        }
      }
      if (!all_zero && item.size() >= tokenizer_.semantic_token_ids.size()) {
        history.push_back(std::move(item));
      }
    }
    if (history.empty()) {
      *error = "empty request history";
      return false;
    }

    while (true) {
      std::vector<int> built;
      std::string build_error;
      if (!BuildPromptTokensForHistory(history, &built, &build_error)) {
        *error = build_error;
        return false;
      }
      if (static_cast<int>(built.size()) <= config_.trt_max_input_len ||
          history.size() <= 1) {
        if (static_cast<int>(built.size()) > config_.trt_max_input_len) {
          built.erase(built.begin(),
                      built.begin() + (built.size() - config_.trt_max_input_len));
        }
        *prompt_tokens = std::move(built);
        return true;
      }
      history.erase(history.begin());
    }
  }

  bool BuildPromptTokensForHistory(
      const std::vector<std::vector<int>>& history,
      std::vector<int>* prompt_tokens,
      std::string* error) const {
    prompt_tokens->clear();
    prompt_tokens->insert(prompt_tokens->end(),
                          tokenizer_.prefix_tokens.begin(),
                          tokenizer_.prefix_tokens.end());
    for (size_t i = 0; i < history.size(); ++i) {
      if (i != 0) {
        prompt_tokens->insert(prompt_tokens->end(),
                              tokenizer_.separator_tokens.begin(),
                              tokenizer_.separator_tokens.end());
      }
      const auto& item = history[i];
      for (size_t layer = 0; layer < tokenizer_.semantic_token_ids.size(); ++layer) {
        const int value = item[layer];
        if (value < 0 ||
            static_cast<size_t>(value) >= tokenizer_.semantic_token_ids[layer].size()) {
          *error = "semantic id value out of tokenizer range";
          return false;
        }
        const int token_id = tokenizer_.semantic_token_ids[layer][value];
        if (token_id < 0) {
          *error = "missing semantic token id in C++ tokenizer config";
          return false;
        }
        prompt_tokens->push_back(token_id);
      }
    }
    prompt_tokens->insert(prompt_tokens->end(),
                          tokenizer_.suffix_tokens.begin(),
                          tokenizer_.suffix_tokens.end());
    return true;
  }

  bool RunExecutor(
      const std::vector<int>& prompt_tokens,
      uint64_t seed,
      std::vector<int>* output_tokens,
      double* elapsed_ms,
      std::string* error) {
    namespace texec = tensorrt_llm::executor;
    butil::Timer timer;
    timer.start();
    try {
      texec::SamplingConfig sampling_config(1);
      if (config_.trt_top_k > 0) {
        sampling_config.setTopK(
            std::optional<texec::SizeType32>(
                static_cast<texec::SizeType32>(config_.trt_top_k)));
      }
      if (config_.trt_temperature > 0.0) {
        sampling_config.setTemperature(
            std::optional<texec::FloatType>(
                static_cast<texec::FloatType>(config_.trt_temperature)));
      }
      sampling_config.setSeed(
          std::optional<texec::RandomSeedType>(
              static_cast<texec::RandomSeedType>(seed)));
      texec::OutputConfig output_config(false, false, false, true);
      texec::Request executor_request(
          texec::VecTokens(prompt_tokens.begin(), prompt_tokens.end()),
          static_cast<texec::SizeType32>(config_.trt_max_new_tokens),
          false,
          sampling_config,
          output_config,
          std::optional<texec::SizeType32>(
              static_cast<texec::SizeType32>(tokenizer_.eos_id)),
          std::optional<texec::SizeType32>(
              static_cast<texec::SizeType32>(tokenizer_.pad_id)));
      const auto request_id = executor_->enqueueRequest(executor_request);
      const auto deadline = std::chrono::steady_clock::now() +
                            std::chrono::milliseconds(config_.trt_request_timeout_ms);

      while (std::chrono::steady_clock::now() < deadline) {
        auto responses =
            executor_->awaitResponses(
                request_id,
                std::optional<std::chrono::milliseconds>(
                    std::chrono::milliseconds(10)));
        for (const auto& response : responses) {
          if (response.hasError()) {
            *error = "TensorRT-LLM Executor error: " + response.getErrorMsg();
            executor_->cancelRequest(request_id);
            return false;
          }
          const auto& result = response.getResult();
          if (result.isFinal) {
            output_tokens->clear();
            if (!result.outputTokenIds.empty()) {
              output_tokens->assign(result.outputTokenIds[0].begin(),
                                    result.outputTokenIds[0].end());
            }
            timer.stop();
            *elapsed_ms = timer.m_elapsed();
            return true;
          }
        }
      }
      executor_->cancelRequest(request_id);
      *error = "TensorRT-LLM Executor request timed out";
      return false;
    } catch (const std::exception& e) {
      *error = std::string("TensorRT-LLM Executor exception: ") + e.what();
      return false;
    }
  }

  std::vector<std::vector<int>> ParseOutputTokens(const std::vector<int>& token_ids) const {
    std::vector<std::vector<int>> layer_values(tokenizer_.semantic_token_ids.size());
    std::vector<std::unordered_set<int>> seen_values(tokenizer_.semantic_token_ids.size());
    for (int token_id : token_ids) {
      auto it = tokenizer_.id_to_semantic.find(token_id);
      if (it == tokenizer_.id_to_semantic.end()) {
        continue;
      }
      const int layer = it->second.first;
      const int value = it->second.second;
      if (layer < 0 || static_cast<size_t>(layer) >= layer_values.size()) {
        continue;
      }
      if (seen_values[layer].insert(value).second) {
        layer_values[layer].push_back(value);
      }
    }

    for (auto& values : layer_values) {
      if (values.empty()) {
        values.push_back(0);
      }
      std::sort(values.begin(), values.end());
    }

    std::vector<std::vector<int>> combinations;
    std::vector<int> current;
    BuildCartesian(layer_values, 0, &current, &combinations);
    return combinations;
  }

  static void BuildCartesian(
      const std::vector<std::vector<int>>& layers,
      size_t layer,
      std::vector<int>* current,
      std::vector<std::vector<int>>* combinations) {
    if (layer == layers.size()) {
      combinations->push_back(*current);
      return;
    }
    for (int value : layers[layer]) {
      current->push_back(value);
      BuildCartesian(layers, layer + 1, current, combinations);
      current->pop_back();
    }
  }

  void FillRecommendations(
      const pairec::inference::RecommendRequest& request,
      const std::vector<std::vector<int>>& semantic_candidates,
      pairec::inference::RecommendResponse* response) const {
    std::unordered_set<std::string> history_keys;
    for (const auto& history : request.history()) {
      history_keys.insert(SemanticKey(history));
    }
    std::unordered_set<int> emitted_items;
    const int topk = request.has_topk() && request.topk() > 0 ? request.topk() : 10;
    for (const auto& semantic_id : semantic_candidates) {
      if (response->recommendations_size() >= topk) {
        break;
      }
      const std::string key = SemanticKey(semantic_id);
      if (history_keys.find(key) != history_keys.end()) {
        continue;
      }
      auto it = semantic_to_candidate_.find(key);
      if (it == semantic_to_candidate_.end()) {
        continue;
      }
      const Candidate* candidate = it->second;
      if (!emitted_items.insert(candidate->item_id).second) {
        continue;
      }
      auto* rec = response->add_recommendations();
      rec->set_item_id(candidate->item_id);
      rec->set_score(1.0);
      for (int value : candidate->semantic_id) {
        rec->add_semantic_id(value);
      }
    }
  }

#if defined(PAIREC_ENABLE_DATASYSTEM_KV_PROBE) && PAIREC_ENABLE_DATASYSTEM_KV_PROBE
  bool InitDatasystemSetGetProbe(std::string* error) {
    if (config_.kvc_probe_object_count < 1) {
      *error = "kvc_probe_object_count must be >= 1";
      return false;
    }
    if (config_.kvc_probe_object_bytes == 0) {
      *error = "kvc_probe_object_bytes must be > 0";
      return false;
    }

    const char* env_host = std::getenv("DATASYSTEM_HOST");
    const char* env_port = std::getenv("DATASYSTEM_PORT");
    if (config_.datasystem_host.empty() && env_host != nullptr) {
      config_.datasystem_host = env_host;
    }
    if (config_.datasystem_port <= 0 && env_port != nullptr) {
      config_.datasystem_port = std::atoi(env_port);
    }
    if (config_.datasystem_host.empty()) {
      config_.datasystem_host = "127.0.0.1";
    }
    if (config_.datasystem_port <= 0) {
      config_.datasystem_port = 31501;
    }

    datasystem::ConnectOptions options;
    options.host = config_.datasystem_host;
    options.port = config_.datasystem_port;
    options.enableCrossNodeConnection = true;
    datasystem_probe_client_.reset(new datasystem::KVClient(options));
    datasystem::Status init_status = datasystem_probe_client_->Init();
    if (init_status.IsError()) {
      *error = "failed to initialize trtllm DataSystem Set/Get probe KVClient: " +
               init_status.ToString();
      return false;
    }

    std::cout << "[brpc-inference] trtllm_datasystem_set_get_probe initialized"
              << " host=" << config_.datasystem_host
              << " port=" << config_.datasystem_port
              << " object_count=" << config_.kvc_probe_object_count
              << " object_bytes=" << config_.kvc_probe_object_bytes
              << " ttl_sec=" << config_.kvc_probe_ttl_sec
              << " timeout_ms=" << config_.kvc_probe_timeout_ms
              << std::endl;
    return true;
  }

  static std::string SanitizeKeyComponent(const std::string& input) {
    std::string out;
    out.reserve(std::min<size_t>(input.size(), 96));
    for (char ch : input) {
      const unsigned char c = static_cast<unsigned char>(ch);
      if (std::isalnum(c) || ch == '~' || ch == '!' || ch == '@' ||
          ch == '#' || ch == '$' || ch == '%' || ch == '^' || ch == '&' ||
          ch == '*' || ch == '.' || ch == '-' || ch == '_') {
        out.push_back(ch);
      } else {
        out.push_back('_');
      }
      if (out.size() >= 96) {
        break;
      }
    }
    if (out.empty()) {
      out = "empty";
    }
    return out;
  }

  std::vector<std::string> BuildDatasystemProbeKeys(
      const pairec::inference::RecommendRequest& request) {
    const uint64_t seq = datasystem_probe_sequence_.fetch_add(1);
    std::string component = request.request_id();
    if (component.empty()) {
      component = request.user_id();
    }
    component = SanitizeKeyComponent(component);

    std::vector<std::string> keys;
    keys.reserve(static_cast<size_t>(config_.kvc_probe_object_count));
    for (int i = 0; i < config_.kvc_probe_object_count; ++i) {
      std::ostringstream out;
      out << "pairec_trtllm_mget_" << component << "_" << seq << "_" << i;
      keys.push_back(out.str());
    }
    return keys;
  }

  bool RunDatasystemSetGetProbe(
      const pairec::inference::RecommendRequest& request,
      pairec::inference::TraceInfo* trace,
      std::string* error) {
    if (!datasystem_probe_client_) {
      *error = "trtllm DataSystem Set/Get probe was not initialized";
      return false;
    }

    std::vector<std::string> keys = BuildDatasystemProbeKeys(request);

    datasystem::SetParam set_param;
    set_param.writeMode = datasystem::WriteMode::NONE_L2_CACHE_EVICT;
    set_param.ttlSecond = static_cast<uint32_t>(std::max(0, config_.kvc_probe_ttl_sec));
    set_param.existence = datasystem::ExistenceOpt::NONE;
    set_param.cacheType = datasystem::CacheType::MEMORY;

    std::vector<char> payload(static_cast<size_t>(config_.kvc_probe_object_bytes));
    const uint64_t hash = StableRequestHash(request);
    for (size_t i = 0; i < payload.size(); ++i) {
      payload[i] = static_cast<char>((i + hash) & 0xff);
    }

    const uint64_t object_bytes =
        static_cast<uint64_t>(config_.kvc_probe_object_bytes);
    std::vector<uint64_t> sizes(keys.size(), object_bytes);
    std::vector<std::shared_ptr<datasystem::Buffer>> buffers;

    butil::Timer create_timer;
    create_timer.start();
    datasystem::Status create_status =
        datasystem_probe_client_->MCreate(keys, sizes, set_param, buffers);
    create_timer.stop();
    const double create_ms = create_timer.m_elapsed();
    const size_t create_call_count = 1;
    if (create_status.IsError()) {
      *error = "trtllm DataSystem MCreate failed: " + create_status.ToString();
      return false;
    }
    if (buffers.size() != keys.size()) {
      *error = "trtllm DataSystem MCreate returned unexpected buffer count";
      return false;
    }

    butil::Timer fill_timer;
    fill_timer.start();
    for (auto& buffer : buffers) {
      if (!buffer) {
        *error = "trtllm DataSystem MCreate returned null buffer";
        return false;
      }
      datasystem::Status copy_status =
          buffer->MemoryCopy(payload.data(), static_cast<uint64_t>(payload.size()));
      if (copy_status.IsError()) {
        *error = "trtllm DataSystem probe buffer copy failed: " +
                 copy_status.ToString();
        return false;
      }
    }
    fill_timer.stop();
    const double fill_ms = fill_timer.m_elapsed();

    butil::Timer set_timer;
    set_timer.start();
    datasystem::Status set_status = datasystem_probe_client_->MSet(buffers);
    set_timer.stop();
    const double set_ms = set_timer.m_elapsed();
    const size_t set_call_count = 1;
    if (set_status.IsError()) {
      *error = "trtllm DataSystem MSet failed: " + set_status.ToString();
      return false;
    }

    butil::Timer get_timer;
    get_timer.start();
    std::vector<datasystem::Optional<datasystem::Buffer>> out_buffers;
    datasystem::Status get_status =
        datasystem_probe_client_->Get(keys, out_buffers, config_.kvc_probe_timeout_ms);
    get_timer.stop();
    const double get_ms = get_timer.m_elapsed();
    const size_t get_call_count = 1;
    if (get_status.IsError()) {
      *error = "trtllm DataSystem MGet failed: " + get_status.ToString();
      return false;
    }

    size_t found_count = 0;
    uint64_t found_bytes = 0;
    for (const auto& buffer : out_buffers) {
      if (!buffer) {
        continue;
      }
      ++found_count;
      found_bytes += static_cast<uint64_t>(buffer->GetSize());
    }
    if (found_count != keys.size()) {
      std::ostringstream out;
      out << "trtllm DataSystem MGet returned " << found_count << "/"
          << keys.size() << " buffers";
      *error = out.str();
      return false;
    }

    const double kv_write_ms = create_ms + fill_ms + set_ms;
    trace->set_kv_write_ms(kv_write_ms);
    trace->set_kv_lookup_ms(get_ms);
    trace->set_kv_source("trtllm_cpp_datasystem_mset_mget");

    std::cout << "[brpc-inference] method=TrtllmDatasystemMSetMGet"
              << " request_id=" << request.request_id()
              << " user=" << request.user_id()
              << " object_count=" << keys.size()
              << " object_bytes=" << config_.kvc_probe_object_bytes
              << " total_bytes=" << found_bytes
              << " create_call_count=" << create_call_count
              << " mcreate_call_count=" << create_call_count
              << " set_call_count=" << set_call_count
              << " get_call_count=" << get_call_count
              << " mset_call_count=" << set_call_count
              << " mget_call_count=" << get_call_count
              << " set_buffer_count=" << buffers.size()
              << " get_key_count=" << keys.size()
              << " out_buffer_count=" << out_buffers.size()
              << " create_ms=" << create_ms
              << " fill_ms=" << fill_ms
              << " set_ms=" << set_ms
              << " get_ms=" << get_ms
              << " found=" << found_count
              << " backend=trtllm_cpp"
              << std::endl;
    return true;
  }
#endif

  ServerConfig config_;
  CppTokenizerConfig tokenizer_;
  std::vector<Candidate> candidates_;
  std::unordered_map<std::string, const Candidate*> semantic_to_candidate_;
  std::unique_ptr<tensorrt_llm::executor::Executor> executor_;
  std::atomic<uint64_t> seed_{42};
#if defined(PAIREC_ENABLE_DATASYSTEM_KV_PROBE) && PAIREC_ENABLE_DATASYSTEM_KV_PROBE
  std::unique_ptr<datasystem::KVClient> datasystem_probe_client_;
  std::atomic<uint64_t> datasystem_probe_sequence_{1};
#endif
#else
  bool Init(const ServerConfig&, std::string* error) override {
    *error = "trtllm_cpp backend is not linked yet; build the TensorRT-LLM C++ "
             "Executor adapter inside the ARM TRT-LLM runtime image with "
             "-DPAIREC_ENABLE_TRTLLM_CPP=ON";
    return false;
  }

  void Recommend(
      const pairec::inference::RecommendRequest&,
      pairec::inference::RecommendResponse*) override {}

  std::string Name() const override {
    return "trtllm_cpp";
  }
#endif
};

std::unique_ptr<InferenceBackend> CreateBackend(const std::string& name) {
  if (name == "semantic_map") {
    return std::unique_ptr<InferenceBackend>(new SemanticMapBackend());
  }
  if (name == "trtllm_cpp") {
    return std::unique_ptr<InferenceBackend>(new TrtllmCppBackend());
  }
  if (name == "datasystem_kv_probe") {
    return std::unique_ptr<InferenceBackend>(new DataSystemKvProbeBackend());
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
    trace->set_request_id(request->request_id());
    const bool exact_datasystem_probe =
        trace->kv_source() == "datasystem_mset_mget_probe" ||
        trace->kv_source() == "trtllm_cpp_datasystem_mset_mget";
    trace->set_datasystem_expected(exact_datasystem_probe || std::getenv("DATASYSTEM_HOST") != nullptr);
    trace->set_datasystem_attribution_complete(exact_datasystem_probe);
    if (exact_datasystem_probe) {
      trace->set_datasystem_sync_get_count(1);
      trace->set_datasystem_sync_set_count(1);
      trace->set_datasystem_sync_get_us(static_cast<int64_t>(trace->kv_lookup_ms() * 1000.0));
      trace->set_datasystem_sync_set_us(static_cast<int64_t>(trace->kv_write_ms() * 1000.0));
      trace->set_datasystem_attribution_reason("request_correlated_explicit_probe");
    } else if (trace->datasystem_expected()) {
      trace->set_datasystem_attribution_reason("native_trt_kvc_request_identity_not_propagated");
    } else {
      trace->set_datasystem_attribution_reason("datasystem_not_expected");
    }

    if (response->recommendations_size() == 0) {
      response->set_code(299);
      if (!response->has_error() || response->error().empty()) {
        response->set_error("items size not enough");
      }
      cntl->SetFailed(response->error());
    }

    std::cout << "[brpc-inference] method=Recommend user=" << request->user_id()
              << " request_id=" << request->request_id()
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

  if (config.trt_plugin_preflight_only) {
    std::string error;
    if (!RunTrtLlmPluginPreflight(&error)) {
      std::cerr << "TensorRT-LLM plugin preflight failed: " << error << std::endl;
      return 1;
    }
    return 0;
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
