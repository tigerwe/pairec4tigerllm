#include <arpa/inet.h>
#include <cerrno>
#include <cstdio>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <netinet/in.h>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

#include <brpc/channel.h>
#include <brpc/controller.h>
#include <google/protobuf/util/json_util.h>

#include "recommend.pb.h"

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

namespace {

struct ProxyConfig {
  int listen_port = 18090;
  std::string brpc_server = "inference:18100";
  int brpc_timeout_ms = 5000;
  int brpc_max_retry = 1;
  size_t max_body_bytes = 4 * 1024 * 1024;
};

struct HttpRequest {
  std::string method;
  std::string path;
  std::map<std::string, std::string> headers;
  std::string body;
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
      << "  --listen_port=18090\n"
      << "  --brpc_server=inference:18100\n"
      << "  --brpc_timeout_ms=5000\n"
      << "  --brpc_max_retry=1\n"
      << "  --max_body_bytes=4194304\n";
}

bool ParseArgs(int argc, char** argv, ProxyConfig* config) {
  for (int i = 1; i < argc; ++i) {
    std::string value;
    if (ConsumeArgValue(argv[i], "listen_port", &value)) {
      config->listen_port = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "brpc_server", &value)) {
      config->brpc_server = value;
    } else if (ConsumeArgValue(argv[i], "brpc_timeout_ms", &value)) {
      config->brpc_timeout_ms = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "brpc_max_retry", &value)) {
      config->brpc_max_retry = std::atoi(value.c_str());
    } else if (ConsumeArgValue(argv[i], "max_body_bytes", &value)) {
      config->max_body_bytes = static_cast<size_t>(std::strtoull(value.c_str(), nullptr, 10));
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

std::string ToLower(std::string value) {
  for (char& ch : value) {
    if (ch >= 'A' && ch <= 'Z') {
      ch = static_cast<char>(ch - 'A' + 'a');
    }
  }
  return value;
}

std::string Trim(const std::string& value) {
  size_t begin = 0;
  while (begin < value.size() && (value[begin] == ' ' || value[begin] == '\t')) {
    ++begin;
  }
  size_t end = value.size();
  while (end > begin &&
         (value[end - 1] == ' ' || value[end - 1] == '\t' || value[end - 1] == '\r')) {
    --end;
  }
  return value.substr(begin, end - begin);
}

std::string JsonEscape(const std::string& input) {
  std::string output;
  output.reserve(input.size() + 8);
  for (char ch : input) {
    switch (ch) {
      case '\\':
        output += "\\\\";
        break;
      case '"':
        output += "\\\"";
        break;
      case '\n':
        output += "\\n";
        break;
      case '\r':
        output += "\\r";
        break;
      case '\t':
        output += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20) {
          char buf[7];
          std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
          output += buf;
        } else {
          output += ch;
        }
    }
  }
  return output;
}

std::string ErrorJson(int code, const std::string& error) {
  std::ostringstream out;
  out << "{\"code\":" << code << ",\"error\":\"" << JsonEscape(error) << "\"}";
  return out.str();
}

bool SendAll(int fd, const std::string& data) {
  const char* ptr = data.data();
  size_t remaining = data.size();
  while (remaining > 0) {
    const ssize_t sent = send(fd, ptr, remaining, MSG_NOSIGNAL);
    if (sent < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    if (sent == 0) {
      return false;
    }
    ptr += sent;
    remaining -= static_cast<size_t>(sent);
  }
  return true;
}

std::string StatusText(int status_code) {
  switch (status_code) {
    case 200:
      return "OK";
    case 400:
      return "Bad Request";
    case 404:
      return "Not Found";
    case 405:
      return "Method Not Allowed";
    case 413:
      return "Payload Too Large";
    case 502:
      return "Bad Gateway";
    case 503:
      return "Service Unavailable";
    default:
      return "Error";
  }
}

void SendHttpResponse(
    int fd,
    int status_code,
    const std::string& body,
    const std::string& content_type = "application/json") {
  std::ostringstream out;
  out << "HTTP/1.1 " << status_code << " " << StatusText(status_code) << "\r\n"
      << "Content-Type: " << content_type << "\r\n"
      << "Content-Length: " << body.size() << "\r\n"
      << "Connection: close\r\n"
      << "\r\n"
      << body;
  SendAll(fd, out.str());
}

bool FindHeaderEnd(const std::string& data, size_t* header_end, size_t* marker_len) {
  size_t pos = data.find("\r\n\r\n");
  if (pos != std::string::npos) {
    *header_end = pos;
    *marker_len = 4;
    return true;
  }
  pos = data.find("\n\n");
  if (pos != std::string::npos) {
    *header_end = pos;
    *marker_len = 2;
    return true;
  }
  return false;
}

bool ParseHeaders(const std::string& header_text, HttpRequest* request, std::string* error) {
  std::istringstream in(header_text);
  std::string line;
  if (!std::getline(in, line)) {
    *error = "missing request line";
    return false;
  }
  if (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }

  std::istringstream request_line(line);
  std::string version;
  request_line >> request->method >> request->path >> version;
  if (request->method.empty() || request->path.empty()) {
    *error = "invalid request line";
    return false;
  }

  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty()) {
      break;
    }
    const size_t colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    request->headers[ToLower(Trim(line.substr(0, colon)))] = Trim(line.substr(colon + 1));
  }
  return true;
}

bool ReadHttpRequest(int fd, size_t max_body_bytes, HttpRequest* request, std::string* error) {
  std::string data;
  char buf[8192];
  size_t header_end = 0;
  size_t marker_len = 0;

  while (!FindHeaderEnd(data, &header_end, &marker_len)) {
    const ssize_t n = recv(fd, buf, sizeof(buf), 0);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      *error = std::string("recv failed: ") + std::strerror(errno);
      return false;
    }
    if (n == 0) {
      *error = "connection closed before headers";
      return false;
    }
    data.append(buf, static_cast<size_t>(n));
    if (data.size() > max_body_bytes + 8192) {
      *error = "request is too large";
      return false;
    }
  }

  if (!ParseHeaders(data.substr(0, header_end), request, error)) {
    return false;
  }

  size_t content_length = 0;
  const auto it = request->headers.find("content-length");
  if (it != request->headers.end() && !it->second.empty()) {
    content_length = static_cast<size_t>(std::strtoull(it->second.c_str(), nullptr, 10));
  }
  if (content_length > max_body_bytes) {
    *error = "request body is too large";
    return false;
  }

  const size_t body_offset = header_end + marker_len;
  while (data.size() < body_offset + content_length) {
    const ssize_t n = recv(fd, buf, sizeof(buf), 0);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      *error = std::string("recv body failed: ") + std::strerror(errno);
      return false;
    }
    if (n == 0) {
      *error = "connection closed before body";
      return false;
    }
    data.append(buf, static_cast<size_t>(n));
  }

  request->body = data.substr(body_offset, content_length);
  return true;
}

class BrpcRecommendClient {
 public:
  bool Init(const ProxyConfig& config) {
    brpc::ChannelOptions options;
    options.protocol = "baidu_std";
    options.connection_type = "pooled";
    options.timeout_ms = config.brpc_timeout_ms;
    options.max_retry = config.brpc_max_retry;

    if (channel_.Init(config.brpc_server.c_str(), "", &options) != 0) {
      return false;
    }
    stub_.reset(new pairec::inference::RecommendService_Stub(&channel_));
    return true;
  }

  bool Recommend(
      const std::string& raw_json,
      const std::string& request_id,
      std::string* response_body,
      std::string* error) {
    pairec::inference::RecommendRequest request;
    request.set_raw_json(raw_json);
    if (!request_id.empty()) {
      request.set_request_id(request_id);
    }

    pairec::inference::RecommendResponse response;
    brpc::Controller cntl;
    stub_->Recommend(&cntl, &request, &response, nullptr);
    if (cntl.Failed()) {
      *error = cntl.ErrorText();
      return false;
    }

    if (!response.raw_json().empty()) {
      *response_body = response.raw_json();
      return true;
    }

    const auto status = google::protobuf::util::MessageToJsonString(response, response_body);
    if (!status.ok()) {
      *error = status.ToString();
      return false;
    }
    return true;
  }

  bool Health(std::string* response_body, std::string* error, bool* healthy) {
    pairec::inference::HealthRequest request;
    pairec::inference::HealthResponse response;
    brpc::Controller cntl;
    stub_->Health(&cntl, &request, &response, nullptr);
    if (cntl.Failed()) {
      *healthy = false;
      *error = cntl.ErrorText();
      return false;
    }

    *healthy = !response.has_code() || response.code() == 200;
    if (!response.raw_json().empty()) {
      *response_body = response.raw_json();
      return true;
    }

    std::ostringstream out;
    out << "{\"code\":" << (response.has_code() ? response.code() : 200)
        << ",\"status\":\"" << JsonEscape(response.status()) << "\"";
    if (response.has_backend()) {
      out << ",\"backend\":\"" << JsonEscape(response.backend()) << "\"";
    }
    out << "}";
    *response_body = out.str();
    return true;
  }

 private:
  brpc::Channel channel_;
  std::unique_ptr<pairec::inference::RecommendService_Stub> stub_;
};

void HandleConnection(int fd, BrpcRecommendClient* client, size_t max_body_bytes) {
  HttpRequest request;
  std::string error;
  if (!ReadHttpRequest(fd, max_body_bytes, &request, &error)) {
    const int status = error.find("large") != std::string::npos ? 413 : 400;
    SendHttpResponse(fd, status, ErrorJson(status, error));
    close(fd);
    return;
  }

  if (request.method == "POST" && request.path == "/recommend") {
    const auto it = request.headers.find("x-request-id");
    const std::string request_id = it == request.headers.end() ? "" : it->second;
    std::string response_body;
    if (!client->Recommend(request.body, request_id, &response_body, &error)) {
      SendHttpResponse(fd, 502, ErrorJson(502, error.empty() ? "brpc Recommend failed" : error));
      close(fd);
      return;
    }
    std::cout << "[brpc-http-proxy] method=Recommend request_bytes=" << request.body.size()
              << " response_bytes=" << response_body.size() << std::endl;
    SendHttpResponse(fd, 200, response_body);
  } else if (request.method == "GET" && (request.path == "/health" || request.path == "/ping")) {
    std::string response_body;
    bool healthy = false;
    if (!client->Health(&response_body, &error, &healthy)) {
      SendHttpResponse(fd, 503, ErrorJson(503, error.empty() ? "brpc Health failed" : error));
      close(fd);
      return;
    }
    std::cout << "[brpc-http-proxy] method=Health healthy=" << (healthy ? 1 : 0)
              << " response_bytes=" << response_body.size() << std::endl;
    SendHttpResponse(fd, healthy ? 200 : 503, response_body);
  } else {
    const bool known_path = request.path == "/recommend" || request.path == "/health" ||
                            request.path == "/ping";
    const int status = known_path ? 405 : 404;
    SendHttpResponse(fd, status, ErrorJson(status, "unsupported path or method"));
  }

  close(fd);
}

}  // namespace

int main(int argc, char** argv) {
  ProxyConfig config;
  if (!ParseArgs(argc, argv, &config)) {
    return 2;
  }
  std::signal(SIGPIPE, SIG_IGN);

  BrpcRecommendClient client;
  if (!client.Init(config)) {
    std::cerr << "Failed to initialize brpc channel to " << config.brpc_server << std::endl;
    return 1;
  }

  const int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd < 0) {
    std::cerr << "socket failed: " << std::strerror(errno) << std::endl;
    return 1;
  }

  int reuse = 1;
  setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(static_cast<uint16_t>(config.listen_port));

  if (bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    std::cerr << "bind failed on port " << config.listen_port
              << ": " << std::strerror(errno) << std::endl;
    close(listen_fd);
    return 1;
  }
  if (listen(listen_fd, 128) != 0) {
    std::cerr << "listen failed: " << std::strerror(errno) << std::endl;
    close(listen_fd);
    return 1;
  }

  std::cout << "brpc http proxy listening on 0.0.0.0:" << config.listen_port
            << ", forwarding to brpc://" << config.brpc_server << std::endl;

  while (true) {
    sockaddr_in peer;
    socklen_t peer_len = sizeof(peer);
    const int fd = accept(listen_fd, reinterpret_cast<sockaddr*>(&peer), &peer_len);
    if (fd < 0) {
      if (errno == EINTR) {
        continue;
      }
      std::cerr << "accept failed: " << std::strerror(errno) << std::endl;
      continue;
    }
    std::thread(HandleConnection, fd, &client, config.max_body_bytes).detach();
  }

  close(listen_fd);
  return 0;
}
