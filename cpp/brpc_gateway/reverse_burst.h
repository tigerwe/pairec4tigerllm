#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace pairec::reverse_burst {

inline constexpr char kControlPrefix[] = "PAIREC_RETURN_CONTROL_V1:";

struct Config {
  std::string endpoint;
  std::string stage;
  int concurrency = 1000;
  int payload_bytes = 102400;
  int marker_timeout_ms = 1000;
  int pressure_timeout_ms = 5000;
  int startup_timeout_ms = 30000;
  bool initially_armed = false;
};

struct Marker {
  std::string request_id;
  int backend_code = 0;
  int item_count = 0;
  std::string response_sha256;
};

struct MarkerResult {
  bool enabled = false;
  bool success = false;
  double wall_ms = 0;
  double sink_ms = 0;
  double front_ms = 0;
  std::string error;
};

class Coordinator {
 public:
  Coordinator();
  ~Coordinator();

  Coordinator(const Coordinator&) = delete;
  Coordinator& operator=(const Coordinator&) = delete;

  bool Init(const Config& config, std::string* error);
  bool Arm(std::string* error);
  bool Disarm(std::string* error);
  bool Trigger(const Marker& marker, MarkerResult* result);

  bool enabled() const;
  bool armed() const;
  bool idle() const;
  int connected_sessions() const;
  int concurrency() const;
  int payload_bytes() const;
  const std::string& stage() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

bool ParseControl(const std::string& payload, std::string* action);
std::string Sha256(const std::string& value);

}  // namespace pairec::reverse_burst
