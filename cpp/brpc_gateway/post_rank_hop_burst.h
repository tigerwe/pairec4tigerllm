#pragma once

#include <memory>
#include <string>

#include "pipeline_service.pb.h"

namespace pairec::post_rank {

struct BurstConfig {
  std::string business_endpoint;
  std::string pressure_endpoint;
  int concurrency = 1000;
  int payload_bytes = 102400;
  int business_timeout_ms = 1000;
  int pressure_timeout_ms = 5000;
  int pressure_start_quorum = 950;
  int pressure_start_timeout_ms = 250;
  int startup_timeout_ms = 120000;
  int startup_batch_size = 64;
  int startup_max_retries = 3;
  int startup_retry_backoff_ms = 100;
};

struct BurstResult {
  bool success = false;
  double business_wall_ms = 0;
  double service_ms = 0;
  double front_brpc_ms = 0;
  double marker_wait_ms = 0;
  int pressure_started_at_business_start = 0;
  std::string error;
};

class BurstCoordinator {
 public:
  BurstCoordinator();
  ~BurstCoordinator();
  BurstCoordinator(const BurstCoordinator&) = delete;
  BurstCoordinator& operator=(const BurstCoordinator&) = delete;

  bool Init(const BurstConfig& config, std::string* error);
  bool Trigger(const pairec::pipeline::RankRequest& request,
               pairec::pipeline::RankResponse* response,
               BurstResult* result);
  bool idle() const;
  int connected_sessions() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

std::string OrderedCandidateSha256(
    const google::protobuf::RepeatedPtrField<pairec::pipeline::RankCandidate>& items);

}  // namespace pairec::post_rank
