#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h>
#include <gz/transport/Node.hh>
#include <gz/msgs.hh>
#include <Eigen/Dense>
#include <argparse/argparse.hpp>
#include <google/protobuf/message.h>
#include <cobs.h>
#include "common.hpp"

using asio::ip::tcp;
static argparse::ArgumentParser args("Michi's Gazebo Proxy");
static asio::io_context io_ctx;
static tcp::acceptor michi_listener(io_ctx);
static tcp::socket michi_socket(io_ctx);

inline size_t
cobs_buffer_size(size_t input_size, bool with_trailing_zero = true)
{
  size_t output_size = input_size + input_size / 254 + 1;
  if (with_trailing_zero)
    output_size++;
  return output_size;
}
void
subscription_callback(const google::protobuf::Message& msg,
                      const gz::transport::MessageInfo& info)
{
  std::string topic_name = info.Topic();
  spdlog::debug("Got message from topic: {}", topic_name);
  int msg_len = msg.ByteSizeLong(), len = msg_len + topic_name.size() + 1;
  std::vector<uint8_t> buffer(len), cobs_buffer(cobs_buffer_size(len));
  std::copy(topic_name.begin(), topic_name.end(), buffer.begin());
  msg.SerializeToArray(buffer.data() + topic_name.size()+1, msg_len);
  if (auto result = cobs_encode(cobs_buffer.data(), cobs_buffer.size(), buffer.data(), buffer.size()); result.status != COBS_ENCODE_OK) {
    spdlog::error("COBS encode failed: {}", int(result.status));
    return;
  }
  auto written = asio::write(michi_socket, asio::buffer(cobs_buffer, cobs_buffer.size()));
  spdlog::debug("Wrote {} bytes on the socket, tag len: {}, payload len: {}", written, topic_name.size()+1, len);
}
int main(int argc, char** argv) {
  args.add_argument("-p", "--port")
    .default_value("6000")
    .help("Port to send subscribed gazebo messages");

  args.add_argument("topics")
    .remaining()
    .help("Gazebo topics to subscribe and republish");

  int log_verbosity = 0;
  args.add_argument("-V", "--verbose")
    .action([&](const auto&) { ++log_verbosity; })
    .append()
    .default_value(false)
    .implicit_value(true)
    .nargs(0);

  try {
    args.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << '\n';
    std::cerr << args;
    return 1;
  }
  switch (log_verbosity) {
    case 0:
    spdlog::set_level(spdlog::level::info);
    break;
    case 1:
    spdlog::info("Verbosity 1: Logging debug messages");
    spdlog::set_level(spdlog::level::debug);
    break;
    default:
    spdlog::info("Verbosity 2: Logging trace messages");
    spdlog::set_level(spdlog::level::trace);
  }
 
  auto endpoint = *tcp::resolver(io_ctx).resolve("0.0.0.0", args.get<std::string>("--port"));
  michi_listener.open(tcp::v4());
  michi_listener.set_option(tcp::acceptor::reuse_address(true));
  michi_listener.bind(endpoint);
  spdlog::info("Waiting on port {} for connection...", args.get<std::string>("--port"));
  michi_listener.listen();
  michi_listener.accept(michi_socket);
  spdlog::info("Connected to client");

  auto topics = args.get<std::vector<std::string>>("topics");
  spdlog::debug("Got {} topics: {}", topics.size(), topics); 
  gz::transport::Node node;
  for (const auto& topic : topics) {
    bool result = node.Subscribe(topic, subscription_callback);
    if (result) continue;
    spdlog::error("Could not subscribe to topic: {}", topic);
  }

  gz::transport::waitForShutdown();
  return 0;
}
