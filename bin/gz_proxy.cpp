#include <gz/transport/MessageInfo.hh>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h>
#include <gz/transport/Node.hh>
#include <gz/msgs.hh>
#include <Eigen/Dense>
#include <argparse/argparse.hpp>
#include <google/protobuf/message.h>

static argparse::ArgumentParser args("Gazebo Proxy");

void
cb(const google::protobuf::Message& msg, const gz::transport::MessageInfo& info)
{
  spdlog::debug("Got message from topic: {}", info.Topic());
}
int main(int argc, char** argv) {
  args.add_argument("-p", "--port")
    .default_value(6000u)
    .help("Port to send subscribed gazebo messages")
    .scan<'i', uint16_t>();

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

  auto topics = args.get<std::vector<std::string>>("topics");
  spdlog::debug("Got {} topics: {}", topics.size(), topics); 

  gz::transport::Node node;
  for (const auto& topic : topics) {
    bool result = node.Subscribe(topic, cb);
    if (result) continue;
    spdlog::error("Could not subscribe to topic: {}");
  }
  gz::transport::waitForShutdown();
  return 0;
}
